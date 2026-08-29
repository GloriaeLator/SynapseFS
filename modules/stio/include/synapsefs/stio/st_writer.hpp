#pragma once
/// \file st_writer.hpp
/// Byte-exact .safetensors writer.
///
/// "Writer" is generous: it concatenates a stored header block with a stream of
/// data-section bytes. That IS the design. We never regenerate a header, because
/// key order is an artifact of the producing writer, __metadata__ differs
/// between producers, and trailing padding is legal and part of the file — a
/// regenerated header was four bytes wrong with every tensor bit-identical.
///
/// The one place a header is actually synthesised is fixture generation, which
/// is Python and lives in fixtures/.

#include <filesystem>
#include <span>

#include <synapsefs/core/error.hpp>
#include <synapsefs/format/manifest.hpp>

namespace sfs::stio {

using core::Result;
using core::Status;

/// Streaming sink for `sfs checkout`: write the header block, then buffer
/// ranges in order, then finish. Verifies the running total against
/// manifest.file.total_bytes and, optionally, the SHA-256 witness.
struct StWriterOptions {
    /// Verify the reconstructed file against manifest.file.sha256 as it is
    /// written. On by default: the check costs one pass we are already making,
    /// and it is the PS's own definition of correct.
    bool verify_sha256 = true;
    int  file_mode     = 0644;
};

class StWriter {
public:
    using Options = StWriterOptions;

    [[nodiscard]] static Result<StWriter> create(const std::filesystem::path& dest,
                                                 const format::Manifest&,
                                                 StWriterOptions = {});
    ~StWriter();
    StWriter(StWriter&&) noexcept;
    StWriter& operator=(StWriter&&) noexcept;
    StWriter(const StWriter&) = delete;
    StWriter& operator=(const StWriter&) = delete;

    /// Append bytes at the current position. Callers write strictly in order.
    [[nodiscard]] Status append(std::span<const std::byte>);

    /// fsync, rename into place, fsync the parent directory, and compare the
    /// computed SHA-256 against the manifest. A mismatch is
    /// ErrKind::HashMismatch and the destination is not created.
    [[nodiscard]] Status finish();

    [[nodiscard]] std::uint64_t written() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    StWriter();
};

}  // namespace sfs::stio
