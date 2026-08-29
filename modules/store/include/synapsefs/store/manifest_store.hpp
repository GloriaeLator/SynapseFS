#pragma once
/// \file manifest_store.hpp
/// Manifest read/write, plus the IObjectSource the reconstructor needs.

#include <memory>

#include <synapsefs/core/interfaces.hpp>
#include <synapsefs/format/manifest.hpp>
#include <synapsefs/store/commit_store.hpp>

namespace sfs::store {

class ManifestStore final : public core::IObjectSource {
public:
    ManifestStore(core::IBlockStore& blocks, CommitStore& commits);
    ~ManifestStore() override;

    [[nodiscard]] Result<format::Manifest> read(const Oid&) const;
    [[nodiscard]] Result<Oid>              write(const format::Manifest&);

    // --- core::IObjectSource -------------------------------------------------
    [[nodiscard]] Result<const format::Manifest*> manifest_for(const Oid& commit) override;
    [[nodiscard]] Result<bool> is_ancestor(const Oid& maybe_ancestor,
                                           const Oid& of) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sfs::store
