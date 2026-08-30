#include <synapsefs/store/manifest_store.hpp>

#include <mutex>
#include <unordered_map>

#include <synapsefs/store/dag.hpp>

namespace sfs::store {

struct ManifestStore::Impl {
    core::IBlockStore* blocks;
    CommitStore* commits;
    std::mutex cache_mu;
    std::unordered_map<Oid, format::Manifest> cache;
};

ManifestStore::ManifestStore(core::IBlockStore& blocks, CommitStore& commits)
    : impl_(std::make_unique<Impl>()) {
    impl_->blocks = &blocks;
    impl_->commits = &commits;
}

ManifestStore::~ManifestStore() = default;

Result<format::Manifest> ManifestStore::read(const Oid& oid) const {
    {
        std::lock_guard lock(impl_->cache_mu);
        auto it = impl_->cache.find(oid);
        if (it != impl_->cache.end()) return it->second;
    }
    auto bytes = impl_->blocks->get(oid, core::ObjectKind::Manifest);
    if (!bytes) return std::unexpected(bytes.error());
    auto m = format::Manifest::parse(*bytes);
    if (!m) return std::unexpected(m.error());

    Oid actual = m->oid();
    if (actual != oid)
        return SFS_ERR(HashMismatch, "manifest does not hash to its own address", oid.to_string());

    {
        std::lock_guard lock(impl_->cache_mu);
        impl_->cache.emplace(oid, *m);
    }
    return *m;
}

Result<Oid> ManifestStore::write(const format::Manifest& m) {
    if (auto st = m.validate(); !st) return std::unexpected(st.error());
    auto bytes = m.to_canonical_json();
    auto oid = impl_->blocks->put(core::ObjectKind::Manifest, bytes);
    if (!oid) return std::unexpected(oid.error());
    {
        std::lock_guard lock(impl_->cache_mu);
        impl_->cache.emplace(*oid, m);
    }
    return oid;
}

Result<const format::Manifest*> ManifestStore::manifest_for(const Oid& commit_oid) {
    auto commit = impl_->commits->get_cached(commit_oid);
    if (!commit) return std::unexpected(commit.error());

    Oid manifest_oid = (*commit)->manifest;
    {
        std::lock_guard lock(impl_->cache_mu);
        auto it = impl_->cache.find(manifest_oid);
        if (it != impl_->cache.end()) return &it->second;
    }
    auto m = read(manifest_oid);
    if (!m) return std::unexpected(m.error());
    std::lock_guard lock(impl_->cache_mu);
    auto [it, _] = impl_->cache.emplace(manifest_oid, *m);
    return &it->second;
}

Result<bool> ManifestStore::is_ancestor(const Oid& maybe_ancestor, const Oid& of) {
    return store::is_ancestor(*impl_->commits, maybe_ancestor, of);
}

}  // namespace sfs::store
