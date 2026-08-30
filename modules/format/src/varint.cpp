#include <synapsefs/format/varint.hpp>

namespace sfs::format {

std::size_t encode_varint(std::uint64_t value, std::span<std::byte> out) noexcept {
    std::size_t n = 0;
    do {
        if (n >= out.size()) return 0;
        std::uint8_t byte = value & 0x7Fu;
        value >>= 7;
        if (value != 0) byte |= 0x80u;
        out[n++] = static_cast<std::byte>(byte);
    } while (value != 0);
    return n;
}

std::size_t decode_varint(std::span<const std::byte> in, std::uint64_t& out) noexcept {
    std::uint64_t result = 0;
    std::size_t i = 0;
    int shift = 0;
    for (; i < in.size() && i < kMaxVarintBytes; ++i) {
        std::uint8_t b = static_cast<std::uint8_t>(in[i]);
        std::uint64_t payload = b & 0x7Fu;

        // Reject an overlong encoding: a final byte of 0 with shift > 0 means
        // this value could have been encoded shorter, and two encodings of the
        // same value must never be allowed to produce the same address.
        if ((b & 0x80u) == 0 && payload == 0 && shift > 0) return 0;

        result |= payload << shift;
        if ((b & 0x80u) == 0) {
            out = result;
            return i + 1;
        }
        shift += 7;
    }
    return 0;  // truncated or too long
}

void append_varint(std::vector<std::byte>& buf, std::uint64_t value) {
    std::byte tmp[kMaxVarintBytes];
    std::size_t n = encode_varint(value, tmp);
    buf.insert(buf.end(), tmp, tmp + n);
}

}  // namespace sfs::format
