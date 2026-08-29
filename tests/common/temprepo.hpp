#pragma once
/// \file temprepo.hpp
/// A repository in a temp directory, removed on destruction.

#include <filesystem>
#include <memory>

#include <synapsefs/core/repo_config.hpp>
#include <synapsefs/store/block_store.hpp>
#include <synapsefs/store/commit_store.hpp>
#include <synapsefs/store/manifest_store.hpp>

namespace sfs::test {

class TempRepo {
public:
    explicit TempRepo(core::RepoConfig cfg = {});
    ~TempRepo();
    TempRepo(const TempRepo&) = delete;
    TempRepo& operator=(const TempRepo&) = delete;

    [[nodiscard]] const core::RepoPaths& paths() const noexcept { return paths_; }
    [[nodiscard]] const std::filesystem::path& root() const noexcept { return paths_.root; }

    [[nodiscard]] core::IBlockStore&    blocks();
    [[nodiscard]] store::RefStore&      refs();
    [[nodiscard]] store::CommitStore&   commits();
    [[nodiscard]] store::ManifestStore& manifests();

    /// Commit a generated checkpoint; returns the new commit id. The tiny
    /// generator lives in harness.hpp so tests do not need fixtures on disk.
    [[nodiscard]] core::Oid commit_checkpoint(const std::filesystem::path& safetensors,
                                              std::string_view message);

    /// Keep the directory around after the test — for debugging a failure.
    void keep() noexcept { keep_ = true; }

private:
    core::RepoPaths paths_;
    bool            keep_ = false;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sfs::test
