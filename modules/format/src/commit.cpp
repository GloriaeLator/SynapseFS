#include <synapsefs/format/commit.hpp>

#include <chrono>
#include <cstdio>
#include <regex>

#include <nlohmann/json.hpp>

namespace sfs::format {

using json = nlohmann::json;

std::string now_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto secs = std::chrono::time_point_cast<std::chrono::seconds>(now);
    std::time_t t = std::chrono::system_clock::to_time_t(secs);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ", tm.tm_year + 1900,
                 tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

bool is_valid_timestamp(std::string_view s) {
    static const std::regex kRe(
        R"(^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$)");
    return std::regex_match(std::string(s), kRe);
}

std::vector<std::byte> Commit::to_canonical_json() const {
    // Sorted keys, no whitespace, no trailing newline — the serialisation IS
    // the address, so nlohmann::json's default key order (insertion order in
    // an ordered_json, or already-sorted in the default json which uses
    // std::map) is relied on here: sfs::format uses json (std::map-backed),
    // which sorts keys lexicographically by construction.
    json j;
    j["format_version"] = format_version;
    json parents_json = json::array();
    for (const auto& p : parents) parents_json.push_back(p.to_string());
    j["parents"] = parents_json;
    j["manifest"] = manifest.to_string();
    j["topology"] = topology.to_string();
    j["timestamp"] = timestamp;
    j["author"] = author;
    j["message"] = message;

    // `json` here is nlohmann's default alias, which is map-backed, so object
    // keys serialise in sorted order regardless of insertion order — that is
    // what makes dump() byte-stable and therefore address-stable.
    std::string s = j.dump();  // no whitespace by default
    std::vector<std::byte> out(s.size());
    std::memcpy(out.data(), s.data(), s.size());
    return out;
}

Oid Commit::oid() const {
    auto bytes = to_canonical_json();
    return core::compute_oid(core::ObjectKind::Commit, bytes);
}

Result<Commit> Commit::parse(std::span<const std::byte> bytes) {
    std::string_view sv(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    json j;
    try {
        j = json::parse(sv);
    } catch (const std::exception& e) {
        return SFS_ERR(MalformedObject, std::string("commit JSON parse failed: ") + e.what());
    }

    Commit c;
    try {
        c.format_version = j.at("format_version").get<std::uint32_t>();
        for (const auto& p : j.at("parents")) {
            auto oid_r = Oid::parse(p.get<std::string>());
            if (!oid_r) return std::unexpected(oid_r.error());
            c.parents.push_back(*oid_r);
        }
        if (c.parents.size() > 2)
            return SFS_ERR(MalformedObject, "commit has more than 2 parents");

        auto manifest_r = Oid::parse(j.at("manifest").get<std::string>());
        if (!manifest_r) return std::unexpected(manifest_r.error());
        c.manifest = *manifest_r;

        auto topo_r = Oid::parse(j.at("topology").get<std::string>());
        if (!topo_r) return std::unexpected(topo_r.error());
        c.topology = *topo_r;

        c.timestamp = j.at("timestamp").get<std::string>();
        c.author = j.value("author", std::string{});
        c.message = j.value("message", std::string{});
    } catch (const std::exception& e) {
        return SFS_ERR(MalformedObject, std::string("commit field missing/malformed: ") + e.what());
    }

    if (!is_valid_timestamp(c.timestamp))
        return SFS_ERR(MalformedObject, "commit timestamp is not RFC 3339 UTC", c.timestamp);

    // A reader that does not verify re-serialisation reproduces the same
    // bytes cannot claim `verify` means anything for JSON objects.
    auto reserialized = c.to_canonical_json();
    if (reserialized.size() != bytes.size() ||
        std::memcmp(reserialized.data(), bytes.data(), bytes.size()) != 0) {
        return SFS_ERR(CanonicalizationMismatch,
                       "commit does not round-trip to identical bytes");
    }

    return c;
}

}  // namespace sfs::format
