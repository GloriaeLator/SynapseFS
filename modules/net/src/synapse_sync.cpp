#include <synapsefs/net/synapse_sync.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <optional>
#include <span>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <nlohmann/json.hpp>

#include <synapsefs/core/oid.hpp>
#include <synapsefs/format/object.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace sfs::net  {

// --- Helper Functions ---

// Untrusted peer input never gets a raw std::stoull/std::stoi: a garbled or
// hostile length/offset/port field must not crash a long-running `sfs serve`
// process. Same class of bug as docs/known-gaps.md's ".synapsefs/config
// parsing" row, in the network-facing sibling of that code.
std::optional<std::uint64_t> safe_stoull(const std::string& s) {
    if (s.empty()) return std::nullopt;
    try {
        std::size_t pos = 0;
        unsigned long long v = std::stoull(s, &pos);
        if (pos != s.size()) return std::nullopt;
        return static_cast<std::uint64_t>(v);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<int> safe_stoi(const std::string& s) {
    if (s.empty()) return std::nullopt;
    try {
        std::size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) return std::nullopt;
        return v;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// Recognises ".../objects/<2-hex>/<62-hex>" -- the loose-object fan-out path
// (docs/storage_format.md) -- and reconstructs the oid it claims to be. Ref/
// HEAD/journal files have no address of this kind and are left alone; the
// caller skips verification for anything this returns nullopt for.
std::optional<core::Oid> object_oid_from_path(const fs::path& p) {
    std::string fname = p.filename().string();
    std::string dname = p.parent_path().filename().string();
    auto is_hex = [](const std::string& s) {
        return !s.empty() && std::all_of(s.begin(), s.end(),
                                         [](unsigned char c) { return std::isxdigit(c) != 0; });
    };
    if (fname.size() != 62 || dname.size() != 2 || !is_hex(fname) || !is_hex(dname))
        return std::nullopt;
    auto oid = core::Oid::parse("b3:" + dname + fname);
    if (!oid) return std::nullopt;
    return *oid;
}

// Decodes the just-received loose-object container and checks that its
// payload's real, framed address (core::compute_oid -- the same primitive
// every writer in this codebase addresses objects with) matches what the
// fan-out path claims. Closes docs/known-gaps.md's "network transfer
// integrity" row: this module used to compare file SIZE only and never
// checked received bytes against their claimed identity before writing them
// into place.
bool verify_received_object_payload(const fs::path& temp_path, const core::Oid& expected) {
    std::ifstream f(temp_path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    auto size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::byte> bytes(size);
    if (!f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)))
        return false;

    auto hdr = format::ObjectHeader::decode(bytes);
    if (!hdr) return false;
    if (hdr->payload_offset() + hdr->payload_len > bytes.size()) return false;

    std::span<const std::byte> payload(bytes.data() + hdr->payload_offset(),
                                       static_cast<std::size_t>(hdr->payload_len));
    return core::compute_oid(hdr->kind, payload) == expected;
}

// Scans .synapsefs for files and maps their paths to current file sizes
json get_local_inventory() {
    json inventory = json::object();
    const std::vector<std::string> target_dirs = {
        ".synapsefs/objects", 
        ".synapsefs/refs", 
        ".synapsefs/ref",
        ".synapsefs/journal"
    };
    
    for (const auto& dir : target_dirs) {
        if (fs::exists(dir)) {
            for (const auto& entry : fs::recursive_directory_iterator(dir)) {
                if (entry.is_regular_file()) {
                    // Force UNIX-style separators so path.find() is 100% reliable
                    std::string path = entry.path().generic_string();
                    
                    if (path.find("/ref") != std::string::npos || path.find("HEAD") != std::string::npos) {
                        std::ifstream file(path, std::ios::binary);
                        if (file) {
                            // Read exact binary contents (including newlines) instead of just the first word
                            std::stringstream buffer;
                            buffer << file.rdbuf();
                            inventory[path] = buffer.str(); 
                        }
                    } else {
                        inventory[path] = fs::file_size(entry);
                    }
                }
            }
        }
    }
    return inventory;
}

// Ensure full buffer transmission
bool send_all(int socket, const char* buffer, size_t length) {
    size_t total_sent = 0;
    while (total_sent < length) {
        ssize_t sent = send(socket, buffer + total_sent, length - total_sent, 0);
        if (sent <= 0) return false;
        // sent > 0 is already checked above, so this narrowing is provably safe.
        total_sent += static_cast<size_t>(sent);
    }
    return true;
}

// Read exactly N bytes (used for JSON payloads)
bool read_exact(int socket, char* buffer, size_t length) {
    size_t total_read = 0;
    while (total_read < length) {
        ssize_t r = recv(socket, buffer + total_read, length - total_read, 0);
        if (r <= 0) return false;
        total_read += static_cast<size_t>(r);
    }
    return true;
}

// Read until a newline delimiter
std::string read_line(int socket) {
    std::string line;
    char c;
    while (recv(socket, &c, 1, 0) > 0) {
        if (c == '\n') break;
        line += c;
    }
    return line;
}

int connect_to_remote(const std::string& url) {
    // Expects bare "ip:port" (see synapse_sync.hpp). The CLI --help text used
    // to say "http://ip:port" -- wrong, and a scheme-prefixed URL used to
    // crash the process uncaught (docs/known-gaps.md's "push/pull URL
    // format" row: find(':') matched the FIRST colon, giving ip="http" and
    // an unparseable port). The help text is now fixed at the source
    // (apps/sfs/cmd/push.cpp/pull.cpp), but this still tolerates a
    // copy-pasted "scheme://" prefix defensively, and never throws on
    // garbage input either way.
    std::string cleaned = url;
    if (auto scheme_end = cleaned.find("://"); scheme_end != std::string::npos) {
        cleaned = cleaned.substr(scheme_end + 3);
    }

    std::size_t colon_pos = cleaned.find(':');
    if (colon_pos == std::string::npos) return -1;

    std::string ip = cleaned.substr(0, colon_pos);
    auto port = safe_stoi(cleaned.substr(colon_pos + 1));
    if (!port || *port < 0 || *port > 65535) return -1;

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(static_cast<std::uint16_t>(*port));
    if (inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr) != 1) {
        close(sock);
        return -1;
    }

    if (connect(sock, reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr)) < 0) {
        close(sock);
        return -1;
    }
    return sock;
}

// --- Protocol IO ---

void send_inventory(int sock, const json& inv) {
    std::string serialized = inv.dump();
    std::string header = std::to_string(serialized.length()) + "\n";
    send_all(sock, header.c_str(), header.length());
    send_all(sock, serialized.c_str(), serialized.length());
}

json receive_inventory(int sock) {
    std::string len_str = read_line(sock);
    auto len = safe_stoull(len_str);
    if (!len) return json::object();

    std::vector<char> buffer(*len);
    if (!read_exact(sock, buffer.data(), *len)) return json::object();

    std::string json_str(buffer.begin(), buffer.end());
    try {
        return json::parse(json_str);
    } catch (const json::parse_error&) {
        // A malformed peer payload is a protocol error, not a crash: same
        // "no exceptions cross a module boundary" contract error.hpp asks of
        // production code, applied to this file's json::parse calls.
        return json::object();
    }
}

// Sends a specific file starting from a requested byte offset
void handle_send_file(int sock, const std::string& filepath, size_t offset) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) return;
    
    std::streamoff tell = file.tellg();
    if (tell < 0) return;  // tellg() failed; nothing sane to send
    size_t total_size = static_cast<size_t>(tell);
    if (offset > total_size) offset = total_size;

    std::string header = "FILE\n" + filepath + "\n" + std::to_string(total_size) + "\n" + std::to_string(offset) + "\n";
    if (!send_all(sock, header.c_str(), header.length())) return;

    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    char buffer[8192];
    size_t remaining = total_size - offset;
    size_t bytes_sent = 0;

    while (bytes_sent < remaining) {
        file.read(buffer, sizeof(buffer));
        std::streamsize got = file.gcount();
        if (got > 0) {
            size_t read_bytes = static_cast<size_t>(got);
            if (!send_all(sock, buffer, read_bytes)) {
                std::cerr << "Transfer dropped while sending: " << filepath << "\n";
                return; // Socket error
            }
            bytes_sent += read_bytes;
        }
    }
}

// Receives a file chunk into a temp file, then atomically renames it upon
// completion. Returns whether the file was fully, correctly received and
// committed -- receiver_sync_loop uses this to report an honest overall
// result instead of always succeeding.
bool handle_receive_file(int sock) {
    std::string filepath = read_line(sock);
    auto total_size_opt = safe_stoull(read_line(sock));
    auto offset_opt = safe_stoull(read_line(sock));
    if (filepath.empty() || !total_size_opt || !offset_opt || *offset_opt > *total_size_opt) {
        std::cerr << "Malformed FILE header from peer; aborting this file.\n";
        return false;
    }
    size_t total_size = *total_size_opt;
    size_t offset = *offset_opt;
    size_t remaining = total_size - offset;

    fs::path final_path(filepath);
    fs::path temp_path = final_path.string() + ".tmp";

    std::error_code ec;
    fs::create_directories(final_path.parent_path(), ec);

    std::ios_base::openmode mode = std::ios::binary;
    mode |= (offset > 0) ? std::ios::app : std::ios::trunc;

    std::ofstream file(temp_path, mode);
    if (!file) {
        std::cerr << "Failed to open temporary file: " << temp_path << "\n";
        return false;
    }

    char buffer[8192];
    size_t bytes_received = 0;
    while (bytes_received < remaining) {
        size_t to_read = std::min(sizeof(buffer), remaining - bytes_received);
        ssize_t r = recv(sock, buffer, to_read, 0);
        if (r <= 0) {
            std::cerr << "Transfer interrupted for " << filepath << ".\n";
            return false;
        }
        file.write(buffer, r);
        bytes_received += static_cast<size_t>(r);
    }
    file.close();

    if (bytes_received != remaining) return false;

    // Integrity check: if this path is a loose object (the fan-out shape
    // under .synapsefs/objects/), verify the bytes we just received actually
    // hash to the address their own path claims BEFORE they ever land at
    // final_path. Ref/HEAD/journal files have no such address and are
    // skipped (object_oid_from_path returns nullopt for them) -- see this
    // file's header comment and docs/known-gaps.md.
    if (auto oid = object_oid_from_path(final_path)) {
        if (!verify_received_object_payload(temp_path, *oid)) {
            std::cerr << "Integrity check FAILED for " << filepath
                      << " -- received bytes do not hash to their claimed object id. "
                         "Discarding; this file was NOT synced.\n";
            std::error_code rm_ec;
            fs::remove(temp_path, rm_ec);
            return false;
        }
    }

    // Explicitly remove existing file to prevent std::filesystem::rename from failing
    if (fs::exists(final_path)) {
        fs::remove(final_path, ec);
    }

    fs::rename(temp_path, final_path, ec);
    if (ec) {
        std::cerr << "Failed to commit " << filepath << ": " << ec.message() << "\n";
        return false;
    }
    std::cout << "Successfully synced: " << filepath << "\n";
    return true;
}

// --- Sync Roles ---

// The receiver queries remote inventory, checks delta, and issues requests
// one by one. Returns whether every requested file was fully received AND
// (for objects) passed its integrity check -- previously this returned
// nothing and callers had no way to know a sync had partially failed.
bool receiver_sync_loop(int sock, const json& remote_inventory) {
    json local_inventory = get_local_inventory();
    bool all_ok = true;

    for (auto it = remote_inventory.begin(); it != remote_inventory.end(); ++it) {
        std::string filepath = it.key();

        if (!local_inventory.contains(filepath) || local_inventory[filepath] != it.value()) {
            fs::path temp_path = filepath + ".tmp";
            size_t offset = 0;

            if (fs::exists(temp_path)) {
                std::error_code ec;
                if (it.value().is_string()) {
                    fs::remove(temp_path, ec); // Refs don't resume, wipe tmp
                } else {
                    size_t remote_size = it.value().get<size_t>();
                    offset = fs::file_size(temp_path);
                    if (offset > remote_size) {
                        fs::remove(temp_path, ec);
                        offset = 0;
                    }
                }
            }

            std::string req = "REQ\n" + filepath + "\n" + std::to_string(offset) + "\n";
            if (!send_all(sock, req.c_str(), req.length())) { all_ok = false; break; }

            std::string cmd = read_line(sock);
            if (cmd == "FILE") {
                if (!handle_receive_file(sock)) all_ok = false;
            } else {
                all_ok = false;  // peer didn't answer the request with a file
            }
        }
    }

    std::string done = "DONE\n";
    send_all(sock, done.c_str(), done.length());
    return all_ok;
}

// The sender waits for specific requests and pushes the requested chunks
void sender_sync_loop(int sock) {
    while (true) {
        std::string cmd = read_line(sock);
        if (cmd == "DONE" || cmd.empty()) break;

        if (cmd == "REQ") {
            std::string filepath = read_line(sock);
            auto offset = safe_stoull(read_line(sock));
            if (filepath.empty() || !offset) {
                std::cerr << "Malformed REQ from peer; ignoring.\n";
                continue;
            }
            handle_send_file(sock, filepath, *offset);
        }
    }
}

// --- Core API ---

bool push(const std::string& remote_url) {
    int sock = connect_to_remote(remote_url);
    if (sock < 0) return false;

    std::string init = "PUSH\n";
    send_all(sock, init.c_str(), init.length());

    // Client is Sender during Push. This protocol has no ack/nack channel
    // back to the sender (docs/known-gaps.md, docs/spec/14-wire-protocol.md),
    // so a push genuinely cannot learn here whether the receiver's own
    // integrity check (verify_received_object_payload, run server-side in
    // serve()'s PUSH branch below) accepted every file -- only that the send
    // loop itself ran without a local socket error. Treat a pushed-to repo as
    // unverified until it runs its own `sfs verify --full`, same guidance
    // threat_model.md already gives.
    json local_inv = get_local_inventory();
    send_inventory(sock, local_inv);
    sender_sync_loop(sock);

    close(sock);
    return true;
}

bool pull(const std::string& remote_url) {
    int sock = connect_to_remote(remote_url);
    if (sock < 0) return false;

    std::string init = "PULL\n";
    send_all(sock, init.c_str(), init.length());

    // Client is Receiver during Pull -- this direction DOES know whether
    // every file it received was intact, since receiver_sync_loop now
    // verifies each object's payload against its claimed address before
    // committing it.
    json remote_inv = receive_inventory(sock);
    bool ok = receiver_sync_loop(sock, remote_inv);

    close(sock);
    return ok;
}

void serve(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (port < 0 || port > 65535) {
        std::cerr << "Invalid port: " << port << "\n";
        close(server_fd);
        return;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(static_cast<std::uint16_t>(port));

    bind(server_fd, reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr));
    listen(server_fd, 5);
    std::cout << "Synapse server listening on port " << port << "...\n";

    while (true) {
        int client_socket = accept(server_fd, nullptr, nullptr);
        if (client_socket < 0) continue;

        std::string command = read_line(client_socket);

        if (command == "PUSH") {
            // Server is Receiver during Push -- this side DOES see integrity
            // failures (unlike the pushing client, see push()'s comment
            // above), so log them even though there is no way to relay them
            // back over this protocol.
            json remote_inv = receive_inventory(client_socket);
            if (!receiver_sync_loop(client_socket, remote_inv)) {
                std::cerr << "Incoming push did not fully verify; the repository may be "
                             "incomplete until the sender retries. Run `sfs verify --full`.\n";
            }
        } else if (command == "PULL") {
            // Server is Sender during Pull
            json local_inv = get_local_inventory();
            send_inventory(client_socket, local_inv);
            sender_sync_loop(client_socket);
        }

        close(client_socket);
    }
}

} // namespace synapse