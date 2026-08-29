#pragma once
/// \file st_header.hpp
/// The safetensors header: parsing it, and preserving it VERBATIM.
///
/// The verbatim part is the point. Regenerating a header from parsed metadata
/// produced a file that was four bytes wrong with every tensor bit-identical:
/// safetensors writes keys in an order that is an artifact of the writer,
/// __metadata__ differs between producers, and trailing padding spaces are
/// legal and part of the bytes. We therefore store the original
/// [8-byte LE length][JSON header] as its own content-addressed block and
/// concatenate. docs/storage_format.md §2.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/tensor.hpp>

namespace sfs::format {

using core::BufferEntry;
using core::Result;
using core::TensorMeta;

/// Parsed view of a safetensors header. The parse is for building the buffer
/// layout; it is NEVER used to regenerate the file.
struct StHeader {
    /// Total size of the [8-byte len][JSON] prefix — the data section starts here.
    std::uint64_t header_extent = 0;

    /// Tensor metadata, keyed by name. Insertion order is not preserved and is
    /// not needed: buffer order comes from the offsets.
    std::unordered_map<std::string, TensorMeta> tensors;

    /// __metadata__ verbatim, for diagnostics only.
    std::unordered_map<std::string, std::string> metadata;

    /// Every tensor, ordered by data offset. This is what the manifest stores,
    /// and it must cover the data section with no gaps and no overlaps.
    [[nodiscard]] std::vector<BufferEntry> buffer_layout() const;
};

/// Parse the header of `file_prefix`, which must contain at least the 8-byte
/// length plus the JSON. Does not read the data section.
[[nodiscard]] Result<StHeader> parse_st_header(std::span<const std::byte> file_prefix);

/// Read the 8-byte little-endian header length only. Enough to know how much to
/// read next, and the first thing both this parser and the diff-artifact reader
/// do — the two formats share the shape on purpose.
[[nodiscard]] Result<std::uint64_t> read_header_len(std::span<const std::byte> first8);

/// Validate the invariants the manifest depends on: entries ordered, contiguous
/// from 0, total matching. docs/spec/10 §4.2.
[[nodiscard]] core::Status validate_buffer_layout(std::span<const BufferEntry> entries,
                                                  std::uint64_t header_extent,
                                                  std::uint64_t total_bytes);

}  // namespace sfs::format
