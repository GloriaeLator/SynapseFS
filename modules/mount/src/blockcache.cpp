/// \file blockcache.cpp
/// Bounded LRU of decompressed frames, with single-flight fill.
/// docs/spec/16-consistency.md §5:
///
///   * Immutability   -> no invalidation, only a fill problem.
///   * Single flight  -> the first arrival fills; the rest wait.
///   * Publication    -> visible only once fully populated, release/acquire.
///   * Eviction       -> a non-zero-refcount entry is never evicted.
///
/// Two lock domains, deliberately not one:
///   - FrameCache::Impl::mu guards the map and the LRU list.
///   - Each entry has its OWN mutex/condvar, used only for the fill
///     handshake (a waiter blocking on a slow decompression of frame A must
///     not also block a concurrent, cheap lookup of frame B).
/// Reader refcounts are a plain atomic on the entry, so that dropping a
/// FrameLease is lock-free and never touches the cache-wide mutex --
/// FrameLease has no access to FrameCache's private Impl (the header grants
/// friendship the other way: FrameCache -> FrameLease), so release must not
/// need it.

#include <synapsefs/mount/blockcache.hpp>

#include <condition_variable>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace sfs::mount {

namespace {
// core::Oid already has a std::hash specialization (oid.hpp); fold the
// tensor/frame indices into it rather than re-hashing the raw digest bytes
// ourselves.
std::size_t hash_combine(std::size_t h, std::size_t v) noexcept {
    return h ^ (v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
}
}  // namespace

std::size_t FrameKeyHash::operator()(const FrameKey& k) const noexcept {
    std::size_t h = std::hash<core::Oid>{}(k.artifact);
    h = hash_combine(h, std::hash<std::uint32_t>{}(k.tensor_index));
    h = hash_combine(h, std::hash<std::uint32_t>{}(k.frame_index));
    return h;
}

// ---------------------------------------------------------------------------
// FrameLease::Entry -- the shared cache slot. Opaque in the header; the full
// definition lives here since only FrameCache and FrameLease need it, and
// both are implemented in this translation unit.
// ---------------------------------------------------------------------------

struct FrameLease::Entry {
    FrameKey                key;
    std::vector<std::byte>  data;

    // Publication state. Set under fill_mutex and observed via acquire loads
    // so that a reader who sees ready==true has a happens-before edge to
    // every write of `data` -- never a half-filled frame.
    std::atomic<bool> ready{false};
    std::atomic<bool> failed{false};
    core::Error        fill_error;

    // Handshake for waiters queued behind an in-flight fill. Deliberately
    // separate from FrameCache::Impl::mu -- see file header comment.
    std::mutex              fill_mutex;
    std::condition_variable fill_cv;

    // Reader refcount. Lock-free: incremented under FrameCache::Impl::mu
    // (alongside the LRU touch, so the two stay consistent from the cache's
    // point of view), decremented lock-free from FrameLease::~FrameLease().
    // An entry with refcount > 0 is pinned; eviction must skip it.
    std::atomic<std::uint32_t> refcount{0};

    // Position in the cache-wide LRU list; valid only while the entry is
    // still present in FrameCache::Impl::lru (guarded by mu).
    std::list<FrameKey>::iterator lru_it;
};

FrameLease::~FrameLease() {
    if (entry_ == nullptr) return;
    // Lock-free on purpose: dropping a lease must never block on I/O or on
    // the cache-wide mutex. The entry becomes eviction-eligible the moment
    // this reaches zero; the next get_or_fill's evict_if_needed() will see
    // it under mu.
    entry_->refcount.fetch_sub(1, std::memory_order_release);
}

FrameLease::FrameLease(FrameLease&& o) noexcept
    : entry_(o.entry_), owner_(o.owner_) {
    o.entry_ = nullptr;
    o.owner_ = nullptr;
}

FrameLease& FrameLease::operator=(FrameLease&& o) noexcept {
    if (this == &o) return *this;
    if (entry_ != nullptr) entry_->refcount.fetch_sub(1, std::memory_order_release);
    entry_ = o.entry_;
    owner_ = o.owner_;
    o.entry_ = nullptr;
    o.owner_ = nullptr;
    return *this;
}

std::span<const std::byte> FrameLease::bytes() const noexcept {
    if (entry_ == nullptr || !entry_->ready.load(std::memory_order_acquire)) return {};
    return std::span<const std::byte>(entry_->data);
}

bool FrameLease::valid() const noexcept {
    return entry_ != nullptr && entry_->ready.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// FrameCache::Impl
// ---------------------------------------------------------------------------

struct FrameCache::Impl {
    std::uint64_t budget_bytes = 0;

    // Guards the map, the LRU list, and bytes_resident. Never held across a
    // fill() call or across a wait on an entry's own condvar -- that split is
    // the whole point (see file header comment).
    mutable std::mutex mu;

    std::unordered_map<FrameKey, std::unique_ptr<FrameLease::Entry>, FrameKeyHash> map;
    std::list<FrameKey> lru;  // front = most recently used, back = least
    std::uint64_t bytes_resident = 0;

    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> misses{0};
    std::atomic<std::uint64_t> single_flight_waits{0};
    std::atomic<std::uint64_t> evictions{0};

    explicit Impl(std::uint64_t budget) : budget_bytes(budget) {}

    // Caller holds mu. Moves key to the front of the LRU list.
    void touch(FrameLease::Entry& e) { lru.splice(lru.begin(), lru, e.lru_it); }

    // Caller holds mu. Evicts LRU-ordered, refcount==0, ready entries until
    // resident bytes are back within budget, or every remaining entry is
    // pinned or still filling -- in which case it stops. Under sustained
    // pressure from more concurrent readers than the budget allows, the
    // daemon serves correctly and slowly rather than evicting something in
    // use (docs/spec/16-consistency.md §5).
    void evict_if_needed() {
        auto it = lru.rbegin();
        while (bytes_resident > budget_bytes && it != lru.rend()) {
            auto found = map.find(*it);
            if (found == map.end()) {  // defensive; shouldn't happen
                ++it;
                continue;
            }
            FrameLease::Entry* e = found->second.get();
            if (e->refcount.load(std::memory_order_relaxed) != 0 ||
                !e->ready.load(std::memory_order_relaxed)) {
                ++it;  // pinned or still filling -- try the next LRU-wards entry
                continue;
            }
            const std::uint64_t sz = e->data.size();
            auto to_erase = std::next(it).base();  // base() of reverse_iterator `it`
            lru.erase(to_erase);
            map.erase(found);
            bytes_resident -= sz;
            evictions.fetch_add(1, std::memory_order_relaxed);
            it = lru.rbegin();  // container mutated; restart the scan from the tail
        }
    }
};

FrameCache::FrameCache(std::uint64_t budget_bytes) : impl_(std::make_unique<Impl>(budget_bytes)) {}
FrameCache::~FrameCache() = default;

core::Result<FrameLease> FrameCache::get_or_fill(
    const FrameKey& key, std::uint64_t size_hint,
    const std::function<core::Status(std::span<std::byte>)>& fill) {
    std::unique_lock<std::mutex> lk(impl_->mu);

    auto found = impl_->map.find(key);
    if (found != impl_->map.end()) {
        FrameLease::Entry* e = found->second.get();

        if (e->ready.load(std::memory_order_acquire)) {
            impl_->hits.fetch_add(1, std::memory_order_relaxed);
            e->refcount.fetch_add(1, std::memory_order_relaxed);
            impl_->touch(*e);
            lk.unlock();
            FrameLease lease;
            lease.entry_ = e;
            lease.owner_ = this;
            return lease;
        }

        if (e->failed.load(std::memory_order_acquire)) {
            core::Error err = e->fill_error;
            lk.unlock();
            return std::unexpected(err);
        }

        // In flight: wait on the ENTRY's own condvar, never on the cache
        // mutex, so other keys stay lock-free while this one fills.
        impl_->single_flight_waits.fetch_add(1, std::memory_order_relaxed);
        lk.unlock();

        {
            std::unique_lock<std::mutex> flk(e->fill_mutex);
            e->fill_cv.wait(flk, [&] {
                return e->ready.load(std::memory_order_acquire) ||
                       e->failed.load(std::memory_order_acquire);
            });
        }

        if (e->failed.load(std::memory_order_acquire)) {
            return std::unexpected(e->fill_error);
        }

        // Published while we waited. Re-take the cache mutex just long
        // enough to pin it and refresh LRU order.
        std::lock_guard<std::mutex> lk2(impl_->mu);
        impl_->hits.fetch_add(1, std::memory_order_relaxed);
        e->refcount.fetch_add(1, std::memory_order_relaxed);
        impl_->touch(*e);
        FrameLease lease;
        lease.entry_ = e;
        lease.owner_ = this;
        return lease;
    }

    // Miss: this thread is the single flight for `key`. Insert a
    // not-yet-ready placeholder so any concurrent arrival for the same key
    // finds it above and waits, instead of starting a second decompression.
    impl_->misses.fetch_add(1, std::memory_order_relaxed);

    auto owned = std::make_unique<FrameLease::Entry>();
    FrameLease::Entry* e = owned.get();
    e->key = key;
    e->data.resize(size_hint);

    impl_->lru.push_front(key);
    e->lru_it = impl_->lru.begin();
    impl_->map.emplace(key, std::move(owned));

    lk.unlock();  // fill() can be slow (decompression); never hold mu here.

    core::Status st = fill(std::span<std::byte>(e->data));

    if (!st) {
        core::Error err = st.error();
        {
            std::lock_guard<std::mutex> flk(e->fill_mutex);
            e->fill_error = err;
            e->failed.store(true, std::memory_order_release);
        }
        e->fill_cv.notify_all();

        // Don't leave a permanently-failed placeholder occupying the map: a
        // transient store error (e.g. a since-repaired object) should be
        // retryable on the next call rather than sticky forever.
        {
            std::lock_guard<std::mutex> lk3(impl_->mu);
            auto it2 = impl_->map.find(key);
            if (it2 != impl_->map.end() && it2->second.get() == e) {
                impl_->lru.erase(e->lru_it);
                impl_->map.erase(it2);
            }
        }
        return std::unexpected(err);
    }

    {
        std::lock_guard<std::mutex> flk(e->fill_mutex);
        e->ready.store(true, std::memory_order_release);
    }
    e->fill_cv.notify_all();

    {
        std::lock_guard<std::mutex> lk4(impl_->mu);
        impl_->bytes_resident += e->data.size();
        e->refcount.fetch_add(1, std::memory_order_relaxed);  // this thread's own lease
        impl_->evict_if_needed();
    }

    FrameLease lease;
    lease.entry_ = e;
    lease.owner_ = this;
    return lease;
}

FrameCache::Stats FrameCache::stats() const noexcept {
    std::lock_guard<std::mutex> lk(impl_->mu);
    Stats s;
    s.hits                = impl_->hits.load(std::memory_order_relaxed);
    s.misses               = impl_->misses.load(std::memory_order_relaxed);
    s.single_flight_waits  = impl_->single_flight_waits.load(std::memory_order_relaxed);
    s.evictions            = impl_->evictions.load(std::memory_order_relaxed);
    s.bytes_resident       = impl_->bytes_resident;
    s.bytes_budget         = impl_->budget_bytes;
    return s;
}

}  // namespace sfs::mount
