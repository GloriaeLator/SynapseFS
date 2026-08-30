#include <synapsefs/store/journal.hpp>

#include <cstring>

#include <nlohmann/json.hpp>

#include <synapsefs/core/oid.hpp>
#include <synapsefs/store/refs.hpp>
#include <synapsefs/util/atomic_io.hpp>
#include <synapsefs/util/file.hpp>

namespace fs = std::filesystem;

namespace sfs::store {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// JournalRecord: framed as [4-byte LE payload_len][payload][32-byte BLAKE3
// digest of payload]. Framed and digested so a torn write (crash mid-fwrite)
// is DETECTED as ErrKind::JournalTorn rather than silently replayed with
// garbage fields.
// ---------------------------------------------------------------------------

std::vector<std::byte> JournalRecord::encode() const {
    json j;
    j["format_version"] = format_version;
    j["op"] = (op == JournalOp::Merge) ? "merge" : "pack";
    j["seq"] = seq;
    j["timestamp"] = timestamp;
    j["ref_name"] = ref_name;
    j["ref_old"] = ref_old ? ref_old->to_string() : "";
    j["ref_new"] = ref_new ? ref_new->to_string() : "";
    j["pack_name"] = pack_name;
    json subsumed_json = json::array();
    for (const auto& o : this->subsumed) subsumed_json.push_back(o.to_string());
    j["subsumed"] = subsumed_json;

    std::string body = j.dump();
    std::vector<std::byte> payload(body.size());
    std::memcpy(payload.data(), body.data(), body.size());

    auto digest = core::digest(payload);

    std::vector<std::byte> out;
    out.resize(4 + payload.size() + core::kOidBytes);
    std::uint32_t len = static_cast<std::uint32_t>(payload.size());
    std::memcpy(out.data(), &len, 4);  // host is little-endian (x86_64/ARM64 Linux targets)
    std::memcpy(out.data() + 4, payload.data(), payload.size());
    std::memcpy(out.data() + 4 + payload.size(), digest.data(), core::kOidBytes);
    return out;
}

Result<JournalRecord> JournalRecord::decode(std::span<const std::byte> bytes) {
    if (bytes.size() < 4 + core::kOidBytes)
        return SFS_ERR(JournalTorn, "journal record shorter than framing overhead");

    std::uint32_t len;
    std::memcpy(&len, bytes.data(), 4);
    if (bytes.size() != 4u + len + core::kOidBytes)
        return SFS_ERR(JournalTorn, "journal record length does not match frame");

    auto payload = bytes.subspan(4, len);
    auto stored_digest = bytes.subspan(4 + len, core::kOidBytes);
    auto actual_digest = core::digest(payload);
    if (std::memcmp(actual_digest.data(), stored_digest.data(), core::kOidBytes) != 0)
        return SFS_ERR(JournalTorn, "journal record digest mismatch (torn write)");

    std::string_view sv(reinterpret_cast<const char*>(payload.data()), payload.size());
    json j;
    try {
        j = json::parse(sv);
    } catch (const std::exception& e) {
        return SFS_ERR(JournalTorn, std::string("journal record JSON parse failed: ") + e.what());
    }

    JournalRecord r;
    try {
        r.format_version = j.at("format_version").get<std::uint32_t>();
        r.op = (j.at("op").get<std::string>() == "merge") ? JournalOp::Merge : JournalOp::Pack;
        r.seq = j.at("seq").get<std::uint64_t>();
        r.timestamp = j.at("timestamp").get<std::string>();
        r.ref_name = j.value("ref_name", std::string{});
        std::string old_s = j.value("ref_old", std::string{});
        if (!old_s.empty()) {
            auto o = core::Oid::parse(old_s);
            if (!o) return std::unexpected(o.error());
            r.ref_old = *o;
        }
        std::string new_s = j.value("ref_new", std::string{});
        if (!new_s.empty()) {
            auto o = core::Oid::parse(new_s);
            if (!o) return std::unexpected(o.error());
            r.ref_new = *o;
        }
        r.pack_name = j.value("pack_name", std::string{});
        if (j.contains("subsumed")) {
            for (const auto& s : j.at("subsumed")) {
                auto o = core::Oid::parse(s.get<std::string>());
                if (!o) return std::unexpected(o.error());
                r.subsumed.push_back(*o);
            }
        }
    } catch (const std::exception& e) {
        return SFS_ERR(JournalTorn, std::string("journal record field error: ") + e.what());
    }
    return r;
}

// ---------------------------------------------------------------------------
// Journal
// ---------------------------------------------------------------------------

Journal::Journal(fs::path dir) : dir_(std::move(dir)) {}

namespace {
fs::path record_path(const fs::path& dir, std::uint64_t seq) {
    return dir / ("j." + std::to_string(seq));
}
}  // namespace

Result<std::uint64_t> Journal::begin(const JournalRecord& rec) {
    std::error_code ec;
    fs::create_directories(dir_, ec);

    fs::path p = record_path(dir_, rec.seq);
    auto bytes = rec.encode();
    util::AtomicWriteOptions opts;
    opts.overwrite = true;
    opts.fsync_contents = true;
    opts.fsync_parent = true;
    if (auto r = util::atomic_write(p, bytes, opts); !r)
        return SFS_ERR(Io, "cannot write journal record", p.string());
    return rec.seq;
}

Status Journal::commit(std::uint64_t seq) {
    fs::path p = record_path(dir_, seq);
    std::error_code ec;
    fs::remove(p, ec);
    if (ec) return SFS_ERR(Io, "cannot remove journal record", p.string());
    util::fsync_dir(dir_);
    return {};
}

Result<std::vector<JournalRecord>> Journal::pending() const {
    std::vector<JournalRecord> out;
    std::error_code ec;
    if (!fs::exists(dir_, ec)) return out;
    for (const auto& entry : fs::directory_iterator(dir_, ec)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().filename().string().substr(0, 2) != "j.") continue;
        auto bytes = util::read_file(entry.path());
        if (!bytes) continue;
        auto rec = JournalRecord::decode(*bytes);
        if (!rec) return std::unexpected(rec.error());  // torn: refuse rather than guess
        out.push_back(*rec);
    }
    return out;
}

Status Journal::recover(RefStore& refs) {
    auto records = pending();
    if (!records) return std::unexpected(records.error());

    for (const auto& rec : *records) {
        if (rec.op == JournalOp::Merge) {
            // The merge journal is written BEFORE the ref/HEAD mutation, so
            // recovery replays it: re-attempt the same CAS. If the ref
            // already equals ref_new, the mutation completed and this is a
            // no-op; if it still equals ref_old, finish it now.
            auto current = refs.resolve(rec.ref_name);
            if (current && rec.ref_new && *current == *rec.ref_new) {
                // Already applied; just clear the record.
            } else if (rec.ref_new) {
                if (auto st = refs.update(rec.ref_name, rec.ref_old, *rec.ref_new); !st) {
                    // Race lost to a legitimate concurrent update is fine to
                    // ignore during recovery (someone else finished it); any
                    // other error is real and must propagate.
                    if (st.error().kind != core::ErrKind::RefRaceLost) return st;
                }
            }
        } else {
            // Pack recovery: since packfiles are not implemented in this
            // build (ADR-0006, deferred), no `gc --pack` operation can ever
            // have produced a Pack record in the first place. A Pack record
            // here indicates a repository written by a build that DOES
            // implement packing; refuse rather than guess at half a format
            // we don't speak.
            return SFS_ERR(NotImplemented,
                           "journal contains a Pack record; packfiles are not supported by "
                           "this build",
                           rec.pack_name);
        }

        fs::path p = record_path(dir_, rec.seq);
        std::error_code ec;
        fs::remove(p, ec);
    }
    return {};
}

}  // namespace sfs::store
