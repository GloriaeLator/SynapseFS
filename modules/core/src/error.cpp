#include <synapsefs/core/error.hpp>

namespace sfs::core {

std::string_view to_string(ErrKind k) noexcept {
    switch (k) {
        case ErrKind::Ok: return "ok";

        case ErrKind::Io: return "io error";
        case ErrKind::NoSuchFile: return "no such file";
        case ErrKind::PermissionDenied: return "permission denied";
        case ErrKind::NoSpace: return "no space left on device";
        case ErrKind::Interrupted: return "interrupted";

        case ErrKind::NotARepository: return "not a synapsefs repository";
        case ErrKind::RepositoryLocked: return "repository locked";
        case ErrKind::UnsupportedFormatVersion: return "unsupported format version";
        case ErrKind::RefNotFound: return "ref not found";
        case ErrKind::RefRaceLost: return "ref update race lost";
        case ErrKind::NotFastForward: return "not a fast-forward";

        case ErrKind::ObjectNotFound: return "object not found";
        case ErrKind::ObjectKindMismatch: return "object kind mismatch";
        case ErrKind::MalformedObject: return "malformed object";
        case ErrKind::CanonicalizationMismatch: return "canonicalization mismatch";

        case ErrKind::HashMismatch: return "hash mismatch";
        case ErrKind::ChunkDigestMismatch: return "chunk digest mismatch";
        case ErrKind::FrameDigestMismatch: return "frame digest mismatch";
        case ErrKind::AncestorInvariantViolated: return "ancestor invariant violated";
        case ErrKind::JournalTorn: return "journal record torn";

        case ErrKind::NotSafetensors: return "not a safetensors file";
        case ErrKind::UnsupportedDType: return "unsupported dtype";
        case ErrKind::ShapeMismatch: return "shape mismatch";
        case ErrKind::TensorNotInBufferLayout: return "tensor not in buffer layout";

        case ErrKind::TopologyParse: return "topology parse error";
        case ErrKind::TopologyIncomplete: return "topology incomplete";
        case ErrKind::BlockFactorMismatch: return "block factor mismatch";
        case ErrKind::NotAlignable: return "not alignable";
        case ErrKind::InvalidPermutation: return "invalid permutation";

        case ErrKind::ChainTooDeep: return "delta chain too deep";
        case ErrKind::MergeConflict: return "merge conflict";

        case ErrKind::MountFailed: return "mount failed";
        case ErrKind::ReadOnlyFilesystem: return "read-only filesystem";

        case ErrKind::ProtocolVersion: return "protocol version mismatch";
        case ErrKind::MalformedFrame: return "malformed frame";
        case ErrKind::ConnectionLost: return "connection lost";
        case ErrKind::PeerError: return "peer error";

        case ErrKind::NotImplemented: return "not implemented";
        case ErrKind::Cancelled: return "cancelled";
        case ErrKind::Internal: return "internal error";
    }
    return "unknown error";
}

bool is_integrity(ErrKind k) noexcept {
    switch (k) {
        case ErrKind::HashMismatch:
        case ErrKind::ChunkDigestMismatch:
        case ErrKind::FrameDigestMismatch:
        case ErrKind::AncestorInvariantViolated:
        case ErrKind::JournalTorn:
            return true;
        default:
            return false;
    }
}

int Error::exit_code() const noexcept {
    // docs/spec/15-cli-contract.md §3.
    if (is_integrity()) return 4;
    switch (kind) {
        case ErrKind::Ok: return 0;
        case ErrKind::NotARepository: return 3;
        case ErrKind::MergeConflict: return 5;
        case ErrKind::RepositoryLocked: return 6;
        case ErrKind::NotImplemented: return 7;
        case ErrKind::ProtocolVersion:
        case ErrKind::MalformedFrame:
        case ErrKind::ConnectionLost:
        case ErrKind::PeerError:
            return 8;
        default: return 1;
    }
}

std::string Error::to_string() const {
    std::string s(core::to_string(kind));
    s += ": ";
    s += what;
    if (!context.empty()) {
        s += " (";
        s += context;
        s += ")";
    }
    return s;
}

Error make_error(ErrKind kind, std::string what, std::string context,
                 std::source_location where) {
    Error e;
    e.kind = kind;
    e.what = std::move(what);
    e.context = std::move(context);
    e.where = where;
    return e;
}

Error from_errno(std::error_code ec, std::string context, std::source_location where) {
    ErrKind kind = ErrKind::Io;
    if (ec == std::errc::no_such_file_or_directory) kind = ErrKind::NoSuchFile;
    else if (ec == std::errc::permission_denied) kind = ErrKind::PermissionDenied;
    else if (ec == std::errc::no_space_on_device) kind = ErrKind::NoSpace;
    else if (ec == std::errc::interrupted) kind = ErrKind::Interrupted;

    Error e;
    e.kind = kind;
    e.what = ec.message();
    e.context = std::move(context);
    e.sys_errno = ec.value();
    e.where = where;
    return e;
}

}  // namespace sfs::core
