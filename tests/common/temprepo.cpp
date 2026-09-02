/// \file temprepo.cpp
/// Implements tests/common/temprepo.hpp.
///
/// Deliberately mirrors apps/sfs/cmd/{init,commit}.cpp's real orchestration
/// (repo layout, RepoConfig::save, RefStore::set_head_symbolic, BlockStore::open,
/// CommitStore::commit_and_advance) rather than poking at store internals
/// directly, so a test built on TempRepo is exercising the same setup path a
/// real `sfs init && sfs commit` would take.
///
/// commit_checkpoint() intentionally stays in the commit_full_only path
/// (apps/sfs/cmd/commit.cpp) rather than the aligned/delta path: TempRepo
/// lives in sfs_test_common, which (like tests/byte_identity.cpp and
/// tests/byte_identity_cnn.cpp) does not link synapsefs::align, so nothing
/// here can call align::Matcher. A test that specifically needs delta groups
/// builds a store::plan_commit_groups() call by hand with its own
/// align::MatchReport, the same pattern tests/byte_identity.cpp already
/// uses — TempRepo's job is a plain, always-valid, Full-only commit history
/// to test everything ELSE (tamper detection, crash windows, concurrent
/// readers, sync) against.

#include "temprepo.hpp"

#include <cstdlib>
#include <random>
#include <stdexcept>

#include <synapsefs/format/commit.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/store/refs.hpp>
#include <synapsefs/stio/st_source.hpp>
#include <synapsefs/stio/st_writer.hpp>

namespace sfs::test {

namespace {

// A directory name unlikely to collide across concurrently-running test
// binaries, without pulling in a UUID library for a test-only helper.
std::filesystem::path make_temp_dir() {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    for (int attempt = 0; attempt < 8; ++attempt) {
        auto candidate = std::filesystem::temp_directory_path() /
                        ("sfs-temprepo-" + std::to_string(rng()));
        std::error_code ec;
        if (std::filesystem::create_directory(candidate, ec)) return candidate;
    }
    throw std::runtime_error("TempRepo: could not create a unique temp directory");
}

core::RepoPaths make_repo_layout(const std::filesystem::path& root, const core::RepoConfig& cfg) {
    core::RepoPaths paths{root};
    std::error_code ec;
    std::filesystem::create_directories(paths.objects(), ec);
    std::filesystem::create_directories(paths.tmp(), ec);
    std::filesystem::create_directories(paths.refs_heads(), ec);
    std::filesystem::create_directories(paths.journal(), ec);
    if (ec) throw std::runtime_error("TempRepo: cannot create repository layout: " + ec.message());

    if (auto st = cfg.save(root); !st) {
        throw std::runtime_error("TempRepo: RepoConfig::save failed: " + st.error().to_string());
    }
    return paths;
}

}  // namespace

struct TempRepo::Impl {
    std::unique_ptr<store::BlockStore> blocks;
    store::RefStore                    refs;
    store::CommitStore                 commits;
    store::ManifestStore               manifests;

    Impl(core::RepoPaths paths, std::unique_ptr<store::BlockStore> b)
        : blocks(std::move(b)), refs(std::move(paths)), commits(*blocks, refs),
          manifests(*blocks, commits) {}
};

TempRepo::TempRepo(core::RepoConfig cfg)
    : paths_{make_temp_dir()} {
    paths_ = make_repo_layout(paths_.root, cfg);

    store::RefStore head_refs(paths_);
    if (auto st = head_refs.set_head_symbolic("refs/heads/main"); !st) {
        throw std::runtime_error("TempRepo: set_head_symbolic failed: " + st.error().to_string());
    }

    auto blocks = store::BlockStore::open(paths_, cfg);
    if (!blocks) {
        throw std::runtime_error("TempRepo: BlockStore::open failed: " +
                                 blocks.error().to_string());
    }

    impl_ = std::make_unique<Impl>(paths_, std::move(*blocks));
}

TempRepo::~TempRepo() {
    if (keep_) return;
    std::error_code ec;
    std::filesystem::remove_all(paths_.root, ec);  // best-effort; a leaked temp
                                                    // dir is not worth failing
                                                    // the test's own teardown over
}

core::IBlockStore&    TempRepo::blocks()    { return *impl_->blocks; }
store::RefStore&      TempRepo::refs()      { return impl_->refs; }
store::CommitStore&   TempRepo::commits()   { return impl_->commits; }
store::ManifestStore& TempRepo::manifests() { return impl_->manifests; }

core::Oid TempRepo::commit_checkpoint(const std::filesystem::path& safetensors,
                                      std::string_view message) {
    auto source = stio::StSource::open(safetensors);
    if (!source) {
        throw std::runtime_error("TempRepo::commit_checkpoint: StSource::open failed: " +
                                 source.error().to_string());
    }

    // HEAD must be symbolic (set by the constructor) — resolve the current
    // tip, same compare-and-swap parent-discovery apps/sfs/cmd/commit.cpp uses.
    static constexpr const char* kRefName = "refs/heads/main";
    std::optional<core::Oid> parent;
    if (auto tip = impl_->refs.resolve(kRefName); tip) {
        parent = *tip;
    } else if (tip.error().kind != core::ErrKind::RefNotFound) {
        throw std::runtime_error("TempRepo::commit_checkpoint: refs.resolve failed: " +
                                 tip.error().to_string());
    }

    stio::Sha256Stream sha;
    sha.update((*source)->header_bytes());

    auto header_oid = impl_->blocks->put(core::ObjectKind::Header, (*source)->header_bytes());
    if (!header_oid) {
        throw std::runtime_error("TempRepo::commit_checkpoint: storing header failed: " +
                                 header_oid.error().to_string());
    }

    format::Manifest manifest;
    manifest.file.name = safetensors.filename().string();
    manifest.file.header_block = *header_oid;
    manifest.file.total_bytes = (*source)->total_bytes();

    // Full-only, mirroring apps/sfs/cmd/commit.cpp's commit_full_only(): every
    // tensor is its own Raw-block singleton group. No alignment, no delta —
    // see this file's header comment for why.
    std::vector<std::byte> tbuf;
    for (const auto& entry : (*source)->buffer_layout()) {
        tbuf.resize(static_cast<std::size_t>(entry.nbytes));
        auto n = (*source)->read_raw(entry.off, tbuf);
        if (!n) {
            throw std::runtime_error("TempRepo::commit_checkpoint: read_raw failed: " +
                                     n.error().to_string());
        }
        if (*n != tbuf.size()) {
            throw std::runtime_error("TempRepo::commit_checkpoint: short read on tensor " +
                                     entry.tensor);
        }
        sha.update(tbuf);

        format::BufferEntry be;
        be.tensor = entry.tensor;
        be.off = entry.off;
        be.nbytes = entry.nbytes;
        be.group = entry.tensor;
        manifest.buffer.push_back(be);

        auto block_oid = impl_->blocks->put(core::ObjectKind::Raw, tbuf);
        if (!block_oid) {
            throw std::runtime_error("TempRepo::commit_checkpoint: storing tensor block failed: " +
                                     block_oid.error().to_string());
        }

        format::GroupEntry g;
        g.mode = format::GroupMode::Full;
        g.block = *block_oid;
        g.chain_depth = 0;
        manifest.groups[entry.tensor] = g;
    }
    manifest.file.sha256 = sha.finish_hex();

    if (auto st = manifest.validate(); !st) {
        throw std::runtime_error("TempRepo::commit_checkpoint: built an invalid manifest: " +
                                 st.error().to_string());
    }

    auto manifest_oid = impl_->manifests.write(manifest);
    if (!manifest_oid) {
        throw std::runtime_error("TempRepo::commit_checkpoint: writing manifest failed: " +
                                 manifest_oid.error().to_string());
    }

    // No real topology: TempRepo never plans delta groups (see file header),
    // so the same "{}" placeholder apps/sfs/cmd/commit.cpp stores for a
    // topology-less commit is enough here too.
    static constexpr char kPlaceholderTopology[] = "{}";
    std::vector<std::byte> topo_bytes(
        reinterpret_cast<const std::byte*>(kPlaceholderTopology),
        reinterpret_cast<const std::byte*>(kPlaceholderTopology) + 2);
    auto topology_oid = impl_->blocks->put(core::ObjectKind::Topology, topo_bytes);
    if (!topology_oid) {
        throw std::runtime_error("TempRepo::commit_checkpoint: storing topology failed: " +
                                 topology_oid.error().to_string());
    }

    format::Commit commit;
    if (parent) commit.parents = {*parent};
    commit.manifest = *manifest_oid;
    commit.topology = *topology_oid;
    commit.timestamp = format::now_timestamp();
    commit.author = "temprepo";
    commit.message = std::string(message);

    auto commit_oid = impl_->commits.commit_and_advance(commit, kRefName, parent);
    if (!commit_oid) {
        throw std::runtime_error("TempRepo::commit_checkpoint: commit_and_advance failed: " +
                                 commit_oid.error().to_string());
    }
    return *commit_oid;
}

}  // namespace sfs::test
