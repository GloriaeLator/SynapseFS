#pragma once
/// \file frame.hpp
/// Wire framing: [u32 LE len][u8 type][payload]. docs/spec/14 §2.
///
/// `len` counts the payload only. Maximum payload is 8 MiB; a peer announcing
/// more is a protocol error and the connection closes. Larger objects are split
/// across BLOCK_DATA frames.

#include <cstddef>
#include <cstdint>
#include <span>
#include <optional>
#include <vector>

#include <synapsefs/core/error.hpp>
#include <synapsefs/core/interfaces.hpp>

/// FrameType completes the opaque declaration in core/interfaces.hpp. It lives
/// in `core` because core::ITransport takes one, and `net` cannot define a type
/// that `core` already names — the enumerators belong next to the wire spec,
/// so they are written here and the type is core's.
namespace sfs::core {

enum class FrameType : std::uint8_t {
    Hello      = 0x01,
    HelloAck   = 0x02,
    RefList    = 0x10,
    Have       = 0x11,
    Want       = 0x12,
    BlockHdr   = 0x20,
    BlockData  = 0x21,
    BlockEnd   = 0x22,
    RefUpdate  = 0x30,
    Done       = 0x7e,
    Error      = 0x7f,
};

}  // namespace sfs::core

namespace sfs::net {

using core::FrameType;
using core::Result;
using core::Status;

inline constexpr std::size_t kMaxFramePayload = 8u * 1024 * 1024;
inline constexpr std::size_t kFrameHeaderSize = 5;

[[nodiscard]] std::string_view to_string(FrameType) noexcept;
[[nodiscard]] bool is_known_frame_type(std::uint8_t) noexcept;

void encode_frame_header(FrameType, std::uint32_t payload_len, std::span<std::byte> out);

struct FrameHeader {
    FrameType     type{};
    std::uint32_t payload_len = 0;
};

[[nodiscard]] Result<FrameHeader> decode_frame_header(std::span<const std::byte>);

/// Incremental reader for a stream socket: feed bytes, get whole frames.
/// Bounded by kMaxFramePayload, so a hostile peer cannot make us allocate.
class FrameReader {
public:
    void feed(std::span<const std::byte>);
    [[nodiscard]] Result<std::optional<core::WireFrame>> next();
    [[nodiscard]] std::size_t buffered() const noexcept;

private:
    std::vector<std::byte> buf_;
    std::size_t            consumed_ = 0;
};

}  // namespace sfs::net
