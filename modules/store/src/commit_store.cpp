#include <synapsefs/store/commit_store.hpp>

#include <mutex>
#include <unordered_map>

namespace sfs::store {

struct CommitStore::Impl {
    core::IBlockStore* blocks;
    RefStore* refs;
    std::mutex cache_mu;
    std::unordered_map<Oid, format::Commit> cache;
};

CommitStore::CommitStore(core::IBlockStore& blocks, RefStore& refs)
    : impl_(std::make_shared<Impl>()) {
    impl_->blocks = &blocks;
    impl_->refs = &refs;
}

Result<format::Commit> CommitStore::read(const Oid& oid) const {
    {
        std::lock_guard lock(impl_->cache_mu);
        auto it = impl_->cache.find(oid);
        if (it != impl_->cache.end()) return it->second;
    }
    auto bytes = impl_->blocks->get(oid, core::ObjectKind::Commit);
    if (!bytes) return std::unexpected(bytes.error());
    auto commit = format::Commit::parse(*bytes);
    if (!commit) return std::unexpected(commit.error());

    Oid actual = commit->oid();
    if (actual != oid)
        return SFS_ERR(HashMismatch, "commit does not hash to its own address", oid.to_string());

    {
        std::lock_guard lock(impl_->cache_mu);
        impl_->cache.emplace(oid, *commit);
    }
    return *commit;
}

Result<Oid> CommitStore::write(const format::Commit& c) {
    auto bytes = c.to_canonical_json();
    auto oid = impl_->blocks->put(core::ObjectKind::Commit, bytes);
    if (!oid) return std::unexpected(oid.error());
    {
        std::lock_guard lock(impl_->cache_mu);
        impl_->cache.emplace(*oid, c);
    }
    return oid;
}

Result<Oid> CommitStore::commit_and_advance(const format::Commit& c, std::string_view ref_name,
                                            std::optional<Oid> expected_tip) {
    // Write and fsync the object first (write() -> BlockStore::put already
    // does temp -> fsync -> rename -> fsync parent), THEN CAS the ref. A crash
    // before the ref update leaves an unreferenced object: wasted disk, never
    // a broken repository.
    auto oid = write(c);
    if (!oid) return std::unexpected(oid.error());

    if (auto st = impl_->refs->update(ref_name, expected_tip, *oid); !st)
        return std::unexpected(st.error());

    return oid;
}

Result<const format::Commit*> CommitStore::get_cached(const Oid& oid) {
    {
        std::lock_guard lock(impl_->cache_mu);
        auto it = impl_->cache.find(oid);
        if (it != impl_->cache.end()) return &it->second;
    }
    auto c = read(oid);
    if (!c) return std::unexpected(c.error());
    std::lock_guard lock(impl_->cache_mu);
    auto [it, _] = impl_->cache.emplace(oid, *c);
    return &it->second;
}

}  // namespace sfs::store
