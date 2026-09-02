#include <synapsefs/core/repo_config.hpp>

#include <fstream>
#include <sstream>

#include <synapsefs/core/oid.hpp>

namespace fs = std::filesystem;

namespace sfs::core {

namespace {

// std::stoul/stoull/stod throw std::invalid_argument/std::out_of_range on
// anything that isn't a clean number -- uncaught, that used to mean a
// hand-corrupted (or just hand-edited-wrong) config file crashed the whole
// process instead of producing the clean core::Error every other failure
// path in this codebase returns (docs/known-gaps.md's ".synapsefs/config
// parsing" row). This wraps every numeric field the same way.
template <class T, class Parser>
[[nodiscard]] Result<T> parse_numeric_field(std::string_view key, const std::string& val,
                                            Parser parser) {
    try {
        std::size_t pos = 0;
        T v = parser(val, &pos);
        if (pos != val.size()) {
            return SFS_ERR(Internal, "config: trailing garbage after number", std::string(key));
        }
        return v;
    } catch (const std::exception& e) {
        return SFS_ERR(Internal, "config: malformed numeric value for '" + std::string(key) + "'",
                       e.what());
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// RepoConfig — a flat "key=value" file, one per line. Deliberately not JSON:
// this file is machine-local policy, never hashed or transmitted, so there is
// no canonicalisation requirement to justify JSON's overhead here.
// ---------------------------------------------------------------------------

Result<RepoConfig> RepoConfig::load(const fs::path& repo_root) {
    RepoConfig cfg;
    fs::path path = RepoPaths{repo_root}.dot() / "config";
    std::ifstream f(path);
    if (!f) return cfg;  // no config file yet: defaults are valid

    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        while (!val.empty() && (val.back() == '\r' || val.back() == '\n')) val.pop_back();

        if (key == "format_version") {
            auto v = parse_numeric_field<unsigned long>(
                key, val, [](const std::string& s, std::size_t* p) { return std::stoul(s, p); });
            if (!v) return std::unexpected(v.error());
            cfg.format_version = *v;
        } else if (key == "chunk_bytes") {
            auto v = parse_numeric_field<unsigned long long>(
                key, val, [](const std::string& s, std::size_t* p) { return std::stoull(s, p); });
            if (!v) return std::unexpected(v.error());
            cfg.chunk_bytes = *v;
        } else if (key == "frame_bytes") {
            auto v = parse_numeric_field<unsigned long long>(
                key, val, [](const std::string& s, std::size_t* p) { return std::stoull(s, p); });
            if (!v) return std::unexpected(v.error());
            cfg.frame_bytes = *v;
        } else if (key == "max_chain_depth") {
            auto v = parse_numeric_field<unsigned long>(
                key, val, [](const std::string& s, std::size_t* p) { return std::stoul(s, p); });
            if (!v) return std::unexpected(v.error());
            cfg.max_chain_depth = *v;
        } else if (key == "snapshot_alpha") {
            auto v = parse_numeric_field<double>(
                key, val, [](const std::string& s, std::size_t* p) { return std::stod(s, p); });
            if (!v) return std::unexpected(v.error());
            cfg.snapshot_alpha = *v;
        } else if (key == "compress_raw") {
            cfg.compress_raw = (val == "1" || val == "true");
        } else if (key == "cache_bytes") {
            auto v = parse_numeric_field<unsigned long long>(
                key, val, [](const std::string& s, std::size_t* p) { return std::stoull(s, p); });
            if (!v) return std::unexpected(v.error());
            cfg.cache_bytes = *v;
        } else if (key == "listen") {
            cfg.listen = val;
        }
    }
    if (auto st = cfg.validate(); !st) return std::unexpected(st.error());
    return cfg;
}

Status RepoConfig::save(const fs::path& repo_root) const {
    if (auto st = validate(); !st) return st;

    fs::path path = RepoPaths{repo_root}.dot() / "config";
    std::ostringstream out;
    out << "format_version=" << format_version << "\n"
        << "chunk_bytes=" << chunk_bytes << "\n"
        << "frame_bytes=" << frame_bytes << "\n"
        << "max_chain_depth=" << max_chain_depth << "\n"
        << "snapshot_alpha=" << snapshot_alpha << "\n"
        << "compress_raw=" << (compress_raw ? "1" : "0") << "\n"
        << "cache_bytes=" << cache_bytes << "\n"
        << "listen=" << listen << "\n";

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return SFS_ERR(Io, "cannot write repo config", path.string());
    f << out.str();
    if (!f) return SFS_ERR(Io, "write failed", path.string());
    return {};
}

Status RepoConfig::validate() const {
    if (format_version == 0) return SFS_ERR(UnsupportedFormatVersion, "format_version is 0");
    if (chunk_bytes == 0) return SFS_ERR(Internal, "chunk_bytes must be nonzero");
    if (frame_bytes == 0) return SFS_ERR(Internal, "frame_bytes must be nonzero");
    if (max_chain_depth == 0) return SFS_ERR(Internal, "max_chain_depth must be nonzero");
    if (snapshot_alpha <= 0.0 || snapshot_alpha > 1.0)
        return SFS_ERR(Internal, "snapshot_alpha must be in (0, 1]");
    return {};
}

// ---------------------------------------------------------------------------
// RepoPaths
// ---------------------------------------------------------------------------

fs::path RepoPaths::dot() const { return root / ".synapsefs"; }
fs::path RepoPaths::objects() const { return dot() / "objects"; }
fs::path RepoPaths::pack() const { return dot() / "pack"; }
fs::path RepoPaths::tmp() const { return dot() / "tmp"; }
fs::path RepoPaths::incoming() const { return dot() / "incoming"; }
fs::path RepoPaths::refs_heads() const { return dot() / "refs" / "heads"; }
fs::path RepoPaths::head() const { return dot() / "HEAD"; }
fs::path RepoPaths::journal() const { return dot() / "journal"; }
fs::path RepoPaths::lock() const { return dot() / "index.lock"; }

fs::path RepoPaths::object_path(const Oid& oid) const { return objects() / oid.fanout_path(); }

Result<RepoPaths> RepoPaths::discover(const fs::path& start) {
    std::error_code ec;
    fs::path p = fs::absolute(start, ec);
    if (ec) return SFS_ERR(Io, "cannot resolve path", start.string());

    while (true) {
        std::error_code exists_ec;
        if (fs::exists(p / ".synapsefs", exists_ec)) return RepoPaths{p};
        if (!p.has_parent_path() || p.parent_path() == p)
            return SFS_ERR(NotARepository, "not a synapsefs repository (or any parent)",
                           start.string());
        p = p.parent_path();
    }
}

}  // namespace sfs::core
