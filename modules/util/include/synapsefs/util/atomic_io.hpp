#pragma once
/// \file atomic_io.hpp
/// The single write primitive. Every durable write in SynapseFS goes through
/// here, and the crash-safety argument in docs/adr/0007 is an argument about
/// this file.
///
///   create tmp/<random> (O_CREAT|O_EXCL)
///   write contents
///   fsync file            <- data durable
///   rename -> final path  <- atomic within the filesystem
///   fsync parent dir      <- the rename is durable
///
/// The last step is the one people skip and the one that matters.

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <string_view>
#include <system_error>
#include <vector>

namespace sfs::util {

struct AtomicWriteOptions {
    /// Directory for the temporary file. MUST be on the same filesystem as the
    /// destination, or the rename is not atomic. Defaults to the destination's
    /// parent; the repository passes `.synapsefs/tmp`.
    std::filesystem::path temp_dir;
    int  mode           = 0644;
    bool fsync_contents = true;
    bool fsync_parent   = true;
    /// If false, an existing destination is left alone and the call succeeds.
    /// Content-addressed objects use this: rewriting identical bytes is a no-op.
    bool overwrite = true;
};

/// Write `data` to `dest` atomically. On any error, `dest` is untouched and no
/// partial file is visible under any name a reader looks at.
[[nodiscard]] std::expected<void, std::error_code> atomic_write(
    const std::filesystem::path& dest, std::span<const std::byte> data,
    const AtomicWriteOptions& = {});

/// Same, from several buffers, without concatenating them first. Used by the
/// object writer, whose payload is a container header plus chunk digests plus
/// the body.
[[nodiscard]] std::expected<void, std::error_code> atomic_write_v(
    const std::filesystem::path& dest, std::span<const std::span<const std::byte>> parts,
    const AtomicWriteOptions& = {});

/// Read a whole small file. For objects, use the block store's ranged reads.
[[nodiscard]] std::expected<std::vector<std::byte>, std::error_code> read_file(
    const std::filesystem::path&);

/// Compare-and-swap on a single-line file: succeeds only if the current
/// contents equal `expected`. This is how refs move (docs/spec/11 §4) and what
/// makes `pull` refuse a non-fast-forward instead of silently discarding.
/// An empty `expected` means "must not currently exist".
[[nodiscard]] std::expected<bool, std::error_code> atomic_replace_if(
    const std::filesystem::path& dest, std::string_view expected, std::string_view desired,
    const AtomicWriteOptions& = {});

/// Remove everything in the temp directory. Everything there is garbage by
/// construction: it is never read, only renamed out of.
[[nodiscard]] std::expected<std::size_t, std::error_code> purge_temp_dir(
    const std::filesystem::path& temp_dir);

}  // namespace sfs::util
