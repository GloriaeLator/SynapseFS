#include <synapsefs/store/refs.hpp>

#include <fstream>
#include <sstream>
#include <cstring>
#include <synapsefs/util/atomic_io.hpp>

namespace fs = std::filesystem;

namespace sfs::store {

namespace {

Result<std::string> read_trimmed(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return SFS_ERR(RefNotFound, "ref file does not exist", p.string());
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

Status write_line(const fs::path& p, std::string_view line) {
    std::string data(line);
    data.push_back('\n');
    std::vector<std::byte> bytes(data.size());
    ::memcpy(bytes.data(), data.data(), data.size());
    util::AtomicWriteOptions opts;
    opts.overwrite = true;
    if (auto r = util::atomic_write(p, bytes, opts); !r)
        return SFS_ERR(Io, "cannot write ref", p.string());
    return {};
}

}  // namespace

RefStore::RefStore(core::RepoPaths paths) : paths_(std::move(paths)) {}

Result<Head> RefStore::read_head() const {
    auto content_r = read_trimmed(paths_.head());
    if (!content_r) return SFS_ERR(RefNotFound, "HEAD does not exist", paths_.head().string());
    const std::string& content = *content_r;

    constexpr std::string_view kPrefix = "ref: ";
    Head h;
    if (content.substr(0, kPrefix.size()) == kPrefix) {
        h.symbolic = content.substr(kPrefix.size());
    } else {
        auto oid_r = Oid::parse(content);
        if (!oid_r) return std::unexpected(oid_r.error());
        h.detached = *oid_r;
    }
    return h;
}

Status RefStore::set_head_symbolic(std::string_view ref_name) {
    return write_line(paths_.head(), std::string("ref: ") + std::string(ref_name));
}

Status RefStore::set_head_detached(const Oid& oid) {
    return write_line(paths_.head(), oid.to_string());
}

Result<Oid> RefStore::resolve(std::string_view name_or_oid) const {
    return rev_parse(name_or_oid);
}

Result<Oid> RefStore::rev_parse(std::string_view spec) const {
    if (spec == "HEAD") {
        auto h = read_head();
        if (!h) return std::unexpected(h.error());
        if (h->is_detached()) return *h->detached;
        return rev_parse(*h->symbolic);
    }

    // "refs/heads/<name>" or bare "<name>".
    fs::path ref_path;
    if (spec.substr(0, 11) == "refs/heads/") {
        ref_path = paths_.refs_heads() / std::string(spec.substr(11));
    } else {
        fs::path candidate = paths_.refs_heads() / std::string(spec);
        std::error_code ec;
        if (fs::exists(candidate, ec)) {
            ref_path = candidate;
        }
    }
    if (!ref_path.empty()) {
        std::error_code ec;
        if (fs::exists(ref_path, ec)) {
            auto content = read_trimmed(ref_path);
            if (!content) return std::unexpected(content.error());
            return Oid::parse(*content);
        }
    }

    // Full oid, or an abbreviation resolved by fanout-directory prefix scan.
    if (spec.substr(0, 3) == "b3:") {
        if (spec.size() == 3 + core::kOidHexChars) return Oid::parse(spec);
        std::string_view hex_prefix = spec.substr(3);
        std::error_code ec;
        std::vector<Oid> matches;
        fs::path objects_dir = paths_.objects();
        if (fs::exists(objects_dir, ec)) {
            for (const auto& top : fs::directory_iterator(objects_dir, ec)) {
                if (!top.is_directory()) continue;
                std::string top_name = top.path().filename().string();
                if (hex_prefix.size() >= 2 && top_name != hex_prefix.substr(0, 2)) continue;
                for (const auto& leaf : fs::directory_iterator(top.path(), ec)) {
                    std::string full_hex = top_name + leaf.path().filename().string();
                    if (full_hex.substr(0, hex_prefix.size()) == hex_prefix) {
                        auto oid_r = Oid::parse("b3:" + full_hex);
                        if (oid_r) matches.push_back(*oid_r);
                    }
                }
            }
        }
        if (matches.size() == 1) return matches[0];
        if (matches.empty())
            return SFS_ERR(RefNotFound, "no object matches abbreviation", std::string(spec));
        return SFS_ERR(MalformedObject, "ambiguous abbreviation", std::string(spec));
    }

    return SFS_ERR(RefNotFound, "cannot resolve ref, branch, or oid", std::string(spec));
}

Result<std::vector<Ref>> RefStore::list_heads() const {
    std::vector<Ref> out;
    std::error_code ec;
    fs::path dir = paths_.refs_heads();
    if (!fs::exists(dir, ec)) return out;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        auto content = read_trimmed(entry.path());
        if (!content) continue;
        auto oid_r = Oid::parse(*content);
        if (!oid_r) continue;
        out.push_back(Ref{entry.path().filename().string(), *oid_r});
    }
    return out;
}

Status RefStore::update(std::string_view ref_name, std::optional<Oid> expected,
                        const Oid& desired) {
    // Handle both "main" and "refs/heads/main" formats
    std::string_view branch_name = ref_name;
    if (ref_name.substr(0, 11) == "refs/heads/") {
        branch_name = ref_name.substr(11);
    }
    
    fs::path p = paths_.refs_heads() / std::string(branch_name);
    std::error_code ec;
    bool exists = fs::exists(p, ec);

    if (expected.has_value()) {
        if (!exists) return SFS_ERR(RefRaceLost, "ref no longer exists", std::string(ref_name));
        auto current = read_trimmed(p);
        if (!current) return std::unexpected(current.error());
        if (*current != expected->to_string())
            return SFS_ERR(RefRaceLost, "ref changed concurrently", std::string(ref_name));
    } else if (exists) {
        return SFS_ERR(RefRaceLost, "ref already exists", std::string(ref_name));
    }

    return write_line(p, desired.to_string());
}

Status RefStore::create_branch(std::string_view name, const Oid& at) {
    return update(name, std::nullopt, at);
}

Status RefStore::delete_branch(std::string_view name, bool force) {
    // Handle both "main" and "refs/heads/main" formats
    std::string_view branch_name = name;
    if (name.substr(0, 11) == "refs/heads/") {
        branch_name = name.substr(11);
    }
    
    fs::path p = paths_.refs_heads() / std::string(branch_name);
    std::error_code ec;
    if (!fs::exists(p, ec))
        return SFS_ERR(RefNotFound, "branch does not exist", std::string(name));

    // Full "unless reachable from another ref" reachability requires walking
    // the commit DAG (store::dag), which this module does not depend on to
    // avoid a circular include (dag.hpp depends on commit_store.hpp, refs.hpp
    // is a leaf). The `branch -d` command performs that reachability check
    // before calling delete_branch; force=true skips it deliberately.
    (void)force;

    if (!fs::remove(p, ec))
        return SFS_ERR(Io, "cannot remove branch ref", std::string(name));
    return {};
}

}  // namespace sfs::store
