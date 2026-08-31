#include <synapsefs/codec/compress.hpp>

#include <cstring>
#include <memory>

#include <zstd.h>

namespace sfs::codec {

using core::ErrKind;

std::size_t compress_bound(std::size_t input_size) noexcept {
    return ZSTD_compressBound(input_size);
}

Result<std::vector<std::byte>> compress_frame(std::span<const std::byte> src,
                                              const CompressOptions& opts) {
    // One call = one complete, independently decompressible zstd frame — no
    // streaming context is kept across calls, which is exactly the
    // "decompression must start at an arbitrary frame with no preceding
    // state" contract in compress.hpp's header comment and spec 12 §4.
    std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> cctx(ZSTD_createCCtx(), &ZSTD_freeCCtx);
    if (!cctx) return SFS_ERR(Internal, "zstd: ZSTD_createCCtx failed");

    ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_compressionLevel, opts.level);
    ZSTD_CCtx_setParameter(cctx.get(), ZSTD_c_checksumFlag, opts.checksum ? 1 : 0);

    std::vector<std::byte> out(compress_bound(src.size()));
    const std::size_t written = ZSTD_compress2(cctx.get(), out.data(), out.size(),
                                               src.data(), src.size());
    if (ZSTD_isError(written))
        return SFS_ERR(Internal, "zstd compress failed", ZSTD_getErrorName(written));

    out.resize(written);
    return out;
}

Result<std::size_t> decompress_frame(std::span<const std::byte> src, std::span<std::byte> dst) {
    // Never allocates: this is on the mount's fault path, and dst is a
    // caller-owned buffer sized from the frame index's known decompressed
    // extent. A dst that's too small surfaces as an ordinary zstd error
    // (dstCapacity too small), not a crash.
    const std::size_t n = ZSTD_decompress(dst.data(), dst.size(), src.data(), src.size());
    if (ZSTD_isError(n))
        return SFS_ERR(MalformedObject, "zstd decompress failed", ZSTD_getErrorName(n));
    return n;
}

namespace {

// The tail: total bytes minus the whole elements that fit. Both the split
// and shuffle transforms leave a partial trailing element (if any) verbatim
// at the end rather than padding, so every transform here is an exact,
// same-size, lossless byte permutation with no allocation and no format
// version bump for odd-sized frames.
[[nodiscard]] std::size_t whole_elems(std::size_t total_bytes, std::uint32_t elem_bytes) noexcept {
    return elem_bytes == 0 ? 0 : total_bytes / elem_bytes;
}

}  // namespace

void byteplane_split(std::span<const std::byte> src, std::span<std::byte> dst,
                     std::uint32_t elem_bytes) noexcept {
    if (elem_bytes <= 1) {
        std::memcpy(dst.data(), src.data(), src.size());
        return;
    }
    const std::size_t n = whole_elems(src.size(), elem_bytes);
    for (std::uint32_t plane = 0; plane < elem_bytes; ++plane) {
        std::byte* out = dst.data() + static_cast<std::size_t>(plane) * n;
        for (std::size_t i = 0; i < n; ++i) out[i] = src[i * elem_bytes + plane];
    }
    const std::size_t tail = src.size() - n * elem_bytes;
    if (tail > 0)
        std::memcpy(dst.data() + static_cast<std::size_t>(elem_bytes) * n,
                   src.data() + static_cast<std::size_t>(elem_bytes) * n, tail);
}

void byteplane_join(std::span<const std::byte> src, std::span<std::byte> dst,
                    std::uint32_t elem_bytes) noexcept {
    if (elem_bytes <= 1) {
        std::memcpy(dst.data(), src.data(), src.size());
        return;
    }
    const std::size_t n = whole_elems(dst.size(), elem_bytes);
    for (std::uint32_t plane = 0; plane < elem_bytes; ++plane) {
        const std::byte* in = src.data() + static_cast<std::size_t>(plane) * n;
        for (std::size_t i = 0; i < n; ++i) dst[i * elem_bytes + plane] = in[i];
    }
    const std::size_t tail = dst.size() - n * elem_bytes;
    if (tail > 0)
        std::memcpy(dst.data() + static_cast<std::size_t>(elem_bytes) * n,
                   src.data() + static_cast<std::size_t>(elem_bytes) * n, tail);
}

void bitshuffle(std::span<const std::byte> src, std::span<std::byte> dst,
               std::uint32_t elem_bytes) noexcept {
    if (elem_bytes == 0 || src.empty()) {
        if (!src.empty()) std::memcpy(dst.data(), src.data(), src.size());
        return;
    }
    // Bit-level transpose, in groups of 8 elements: for group g and bit
    // position `bit` within an element, one output byte packs bit `bit` of
    // elements [8g, 8g+8). Only whole groups of 8 are transposed — a
    // trailing partial group (n % 8 elements) plus any partial trailing
    // element are copied verbatim, so the transform never changes size.
    const std::size_t n = whole_elems(src.size(), elem_bytes);
    const std::size_t n8 = (n / 8) * 8;
    const std::size_t total_bits = static_cast<std::size_t>(elem_bytes) * 8;
    const std::size_t bytes_per_plane = n8 / 8;

    for (std::size_t bit = 0; bit < total_bits; ++bit) {
        const std::size_t byte_in_elem = bit / 8;
        const auto bit_in_byte = static_cast<std::uint8_t>(bit % 8);
        std::byte* plane = dst.data() + bit * bytes_per_plane;
        for (std::size_t g = 0; g < bytes_per_plane; ++g) {
            std::uint8_t packed = 0;
            for (std::uint8_t k = 0; k < 8; ++k) {
                const std::size_t i = g * 8 + k;
                const auto byte_val = std::to_integer<std::uint8_t>(src[i * elem_bytes + byte_in_elem]);
                packed = static_cast<std::uint8_t>(packed | (((byte_val >> bit_in_byte) & 1u) << k));
            }
            plane[g] = std::byte{packed};
        }
    }

    const std::size_t shuffled_bytes = total_bits * bytes_per_plane;  // == n8 * elem_bytes
    const std::size_t tail_bytes = src.size() - shuffled_bytes;
    if (tail_bytes > 0)
        std::memcpy(dst.data() + shuffled_bytes, src.data() + shuffled_bytes, tail_bytes);
}

void bitunshuffle(std::span<const std::byte> src, std::span<std::byte> dst,
                  std::uint32_t elem_bytes) noexcept {
    if (elem_bytes == 0 || src.empty()) {
        if (!src.empty()) std::memcpy(dst.data(), src.data(), src.size());
        return;
    }
    const std::size_t n = whole_elems(dst.size(), elem_bytes);
    const std::size_t n8 = (n / 8) * 8;
    const std::size_t total_bits = static_cast<std::size_t>(elem_bytes) * 8;
    const std::size_t bytes_per_plane = n8 / 8;
    const std::size_t shuffled_bytes = total_bits * bytes_per_plane;

    // Every output byte is assembled bit-by-bit across `total_bits` separate
    // passes (one per source bit-plane), so it must start zeroed.
    std::memset(dst.data(), 0, shuffled_bytes);

    for (std::size_t bit = 0; bit < total_bits; ++bit) {
        const std::size_t byte_in_elem = bit / 8;
        const auto bit_in_byte = static_cast<std::uint8_t>(bit % 8);
        const std::byte* plane = src.data() + bit * bytes_per_plane;
        for (std::size_t g = 0; g < bytes_per_plane; ++g) {
            const auto packed = std::to_integer<std::uint8_t>(plane[g]);
            for (std::uint8_t k = 0; k < 8; ++k) {
                const std::size_t i = g * 8 + k;
                const std::uint8_t bitval = (packed >> k) & 1u;
                dst[i * elem_bytes + byte_in_elem] |= std::byte(bitval << bit_in_byte);
            }
        }
    }

    const std::size_t tail_bytes = dst.size() - shuffled_bytes;
    if (tail_bytes > 0)
        std::memcpy(dst.data() + shuffled_bytes, src.data() + shuffled_bytes, tail_bytes);
}

}  // namespace sfs::codec
