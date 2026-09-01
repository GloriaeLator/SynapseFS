/// The race the single-flight design exists to prevent: N threads all fault
/// the same cold frame at once. Exactly one must run `fill`; the rest must
/// block and then observe a fully-published frame, never a torn or
/// half-written one. Run under TSan (the tsan preset exists specifically for
/// this file, per modules/mount/CMakeLists.txt) as well as plain dev.

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include <synapsefs/mount/blockcache.hpp>

using namespace sfs;

namespace {
core::Oid make_oid(std::uint8_t seed) {
    std::vector<std::byte> payload(8, std::byte(seed));
    return core::compute_oid(core::ObjectKind::Raw, payload);
}
}  // namespace

TEST_CASE("N racing threads on one cold key: fill runs exactly once", "[mount][blockcache][concurrency][slow]") {
    constexpr int kThreads = 32;
    constexpr int kRepeats = 200;  // repeat the whole race many times to shake out ordering luck

    for (int rep = 0; rep < kRepeats; ++rep) {
        mount::FrameCache cache(1024 * 1024);
        mount::FrameKey key{make_oid(static_cast<std::uint8_t>(rep)), 0, 0};

        std::atomic<int> fill_calls{0};
        std::atomic<int> ready_threads{0};
        std::atomic<bool> go{false};

        auto fill = [&](std::span<std::byte> buf) -> core::Status {
            fill_calls.fetch_add(1, std::memory_order_relaxed);
            // Give waiters a real window to queue up behind the in-flight
            // fill, rather than each thread racing in and out before any
            // other thread even calls get_or_fill.
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            std::memset(buf.data(), 0x5A, buf.size());
            return {};
        };

        std::vector<std::thread> threads;
        std::atomic<bool> mismatch{false};

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&] {
                ready_threads.fetch_add(1, std::memory_order_relaxed);
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                auto lease = cache.get_or_fill(key, 32, fill);
                if (!lease.has_value() || !lease->valid()) {
                    mismatch.store(true);
                    return;
                }
                for (auto b : lease->bytes()) {
                    if (b != std::byte(0x5A)) {
                        mismatch.store(true);
                        break;
                    }
                }
            });
        }

        while (ready_threads.load(std::memory_order_relaxed) < kThreads) {
            std::this_thread::yield();
        }
        go.store(true, std::memory_order_release);

        for (auto& th : threads) th.join();

        INFO("rep = " << rep);
        CHECK_FALSE(mismatch.load());
        CHECK(fill_calls.load() == 1);
    }
}

TEST_CASE("racing fills on distinct keys under a tight budget never corrupt each other",
         "[mount][blockcache][concurrency][slow]") {
    // Same shape as the single-key race, but every thread has its own frame
    // and the budget forces eviction to interleave with fills -- eviction
    // must only ever touch unheld, ready entries (blockcache.cpp's
    // evict_if_needed), never one a concurrent thread is still filling or
    // holding.
    mount::FrameCache cache(256);  // small: several frames' worth, not all of them
    core::Oid artifact = make_oid(0xEE);

    constexpr int kThreads = 16;
    constexpr int kItersPerThread = 500;
    std::atomic<bool> mismatch{false};
    std::vector<std::thread> threads;

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kItersPerThread; ++i) {
                mount::FrameKey key{artifact, 0, static_cast<std::uint32_t>((t + i) % 40)};
                const std::uint8_t marker = static_cast<std::uint8_t>(key.frame_index);
                auto lease = cache.get_or_fill(key, 8, [marker](std::span<std::byte> buf) {
                    std::memset(buf.data(), marker, buf.size());
                    return core::Status{};
                });
                if (!lease.has_value() || !lease->valid()) {
                    mismatch.store(true);
                    return;
                }
                for (auto b : lease->bytes()) {
                    if (b != std::byte(marker)) {
                        mismatch.store(true);
                        return;
                    }
                }
            }
        });
    }
    for (auto& th : threads) th.join();

    CHECK_FALSE(mismatch.load());
}
