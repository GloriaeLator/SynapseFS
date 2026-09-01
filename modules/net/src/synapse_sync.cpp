#include <synapsefs/net/synapse_sync.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace sfs::net  {

// --- Helper Functions ---

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
        total_sent += sent;
    }
    return true;
}

// Read exactly N bytes (used for JSON payloads)
bool read_exact(int socket, char* buffer, size_t length) {
    size_t total_read = 0;
    while (total_read < length) {
        ssize_t r = recv(socket, buffer + total_read, length - total_read, 0);
        if (r <= 0) return false;
        total_read += r;
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
    size_t colon_pos = url.find(':');
    if (colon_pos == std::string::npos) return -1;
    
    std::string ip = url.substr(0, colon_pos);
    int port = std::stoi(url.substr(colon_pos + 1));

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr);

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
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
    if (len_str.empty()) return json::object();
    
    size_t len = std::stoull(len_str);
    std::vector<char> buffer(len);
    if (!read_exact(sock, buffer.data(), len)) return json::object();
    
    std::string json_str(buffer.begin(), buffer.end());
    return json::parse(json_str);
}

// Sends a specific file starting from a requested byte offset
void handle_send_file(int sock, const std::string& filepath, size_t offset) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) return;
    
    size_t total_size = file.tellg();
    if (offset > total_size) offset = total_size;

    std::string header = "FILE\n" + filepath + "\n" + std::to_string(total_size) + "\n" + std::to_string(offset) + "\n";
    if (!send_all(sock, header.c_str(), header.length())) return;

    file.seekg(offset, std::ios::beg);
    char buffer[8192];
    size_t remaining = total_size - offset;
    size_t bytes_sent = 0;

    while (bytes_sent < remaining) {
        file.read(buffer, sizeof(buffer));
        size_t read_bytes = file.gcount();
        if (read_bytes > 0) {
            if (!send_all(sock, buffer, read_bytes)) {
                std::cerr << "Transfer dropped while sending: " << filepath << "\n";
                return; // Socket error
            }
            bytes_sent += read_bytes;
        }
    }
}

// Receives a file chunk into a temp file, then atomically renames it upon completion
void handle_receive_file(int sock) {
    std::string filepath = read_line(sock);
    size_t total_size = std::stoull(read_line(sock));
    size_t offset = std::stoull(read_line(sock));
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
        return;
    }

    char buffer[8192];
    size_t bytes_received = 0;
    while (bytes_received < remaining) {
        size_t to_read = std::min(sizeof(buffer), remaining - bytes_received);
        ssize_t r = recv(sock, buffer, to_read, 0);
        if (r <= 0) {
            std::cerr << "Transfer interrupted for " << filepath << ".\n";
            return;
        }
        file.write(buffer, r);
        bytes_received += r;
    }
    file.close();

    if (bytes_received == remaining) {
        // Explicitly remove existing file to prevent std::filesystem::rename from failing
        if (fs::exists(final_path)) {
            fs::remove(final_path, ec);
        }
        
        fs::rename(temp_path, final_path, ec);
        if (ec) {
            std::cerr << "Failed to commit " << filepath << ": " << ec.message() << "\n";
        } else {
            std::cout << "Successfully synced: " << filepath << "\n";
        }
    }
}

// --- Sync Roles ---

// The receiver queries remote inventory, checks delta, and issues requests one by one
void receiver_sync_loop(int sock, const json& remote_inventory) {
    json local_inventory = get_local_inventory();

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
            if (!send_all(sock, req.c_str(), req.length())) break;

            std::string cmd = read_line(sock);
            if (cmd == "FILE") {
                handle_receive_file(sock);
            }
        }
    }
    
    std::string done = "DONE\n";
    send_all(sock, done.c_str(), done.length());
}

// The sender waits for specific requests and pushes the requested chunks
void sender_sync_loop(int sock) {
    while (true) {
        std::string cmd = read_line(sock);
        if (cmd == "DONE" || cmd.empty()) break;
        
        if (cmd == "REQ") {
            std::string filepath = read_line(sock);
            size_t offset = std::stoull(read_line(sock));
            handle_send_file(sock, filepath, offset);
        }
    }
}

// --- Core API ---

bool push(const std::string& remote_url) {
    int sock = connect_to_remote(remote_url);
    if (sock < 0) return false;

    std::string init = "PUSH\n";
    send_all(sock, init.c_str(), init.length());

    // Client is Sender during Push
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

    // Client is Receiver during Pull
    json remote_inv = receive_inventory(sock);
    receiver_sync_loop(sock, remote_inv);

    close(sock);
    return true;
}

void serve(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    listen(server_fd, 5);
    std::cout << "Synapse server listening on port " << port << "...\n";

    while (true) {
        int client_socket = accept(server_fd, nullptr, nullptr);
        if (client_socket < 0) continue;

        std::string command = read_line(client_socket);

        if (command == "PUSH") {
            // Server is Receiver during Push
            json remote_inv = receive_inventory(client_socket);
            receiver_sync_loop(client_socket, remote_inv);
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