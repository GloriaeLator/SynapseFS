#include <synapsefs/net/network.hpp>

#include <json-c/json.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace sfs::net {
namespace {

struct Url { 
  std::string host; 
  std::string port; 
};

struct Response { 
  int status; 
  std::string body; 
};

// Parses an HTTP URL into host and port components[cite: 2]
Url parse_url(const std::string& url) {
  constexpr const char* prefix = "http://";
  if (url.rfind(prefix, 0) != 0) {
    throw std::runtime_error("remote URL must start with http://");
  }
  
  auto authority = url.substr(std::strlen(prefix));
  const auto colon = authority.rfind(':');
  
  // Default to port 8080 if not specified[cite: 2]
  if (colon == std::string::npos) {
    return {authority, "8080"};
  }
  return {authority.substr(0, colon), authority.substr(colon + 1)};
}

// Establishes a TCP socket connection to a resolved host[cite: 2]
int connect_socket(const Url& url) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* addresses = nullptr;
  
  if (getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &addresses) != 0) {
    throw std::runtime_error("cannot resolve remote host");
  }
  
  int descriptor = -1;
  for (auto* address = addresses; address; address = address->ai_next) {
    descriptor = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (descriptor >= 0 && connect(descriptor, address->ai_addr, address->ai_addrlen) == 0) {
      break; // Successfully connected[cite: 2]
    }
    if (descriptor >= 0) {
      close(descriptor);
    }
    descriptor = -1;
  }
  freeaddrinfo(addresses);
  
  if (descriptor < 0) {
    throw std::runtime_error("cannot connect to peer");
  }
  return descriptor;
}

// Ensures all bytes are written to the socket[cite: 2]
void send_all(int descriptor, const std::string& bytes) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto sent = send(descriptor, bytes.data() + offset, bytes.size() - offset, 0);
    if (sent <= 0) {
      throw std::runtime_error("network write failed");
    }
    offset += static_cast<std::size_t>(sent);
  }
}

// Reads from the socket until closed[cite: 2]
std::string receive_all(int descriptor) {
  std::string result;
  char buffer[64 * 1024];
  while (true) {
    const auto count = recv(descriptor, buffer, sizeof(buffer), 0);
    if (count == 0) break;
    if (count < 0) throw std::runtime_error("network read failed");
    result.append(buffer, static_cast<std::size_t>(count));
  }
  return result;
}

// Constructs and executes an HTTP 1.1 client request[cite: 2]
Response client_request(const Url& url, const std::string& method, const std::string& path, 
                        const std::string& body = "", const std::string& type = "application/json") {
  const int descriptor = connect_socket(url);
  try {
    std::ostringstream request;
    request << method << " " << path << " HTTP/1.1\r\n"
            << "Host: " << url.host << "\r\n"
            << "Connection: close\r\n";
            
    if (method == "POST" || method == "PUT") {
      request << "Content-Type: " << type << "\r\n"
              << "Content-Length: " << body.size() << "\r\n";
    }
    request << "\r\n" << body;
    
    send_all(descriptor, request.str());
    const auto raw = receive_all(descriptor);
    close(descriptor);
    
    // Parse the HTTP response headers and body[cite: 2]
    const auto header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
      throw std::runtime_error("invalid HTTP response");
    }
    
    const auto first_end = raw.find("\r\n");
    std::istringstream line(raw.substr(0, first_end));
    std::string version; 
    int status{}; 
    line >> version >> status;
    
    return {status, raw.substr(header_end + 4)};
  } catch (...) { 
    close(descriptor); 
    throw; 
  }
}

std::string object_path(const ObjectId& id) { 
  return "/v1/objects/" + kind_name(id.kind) + "/" + id.hash; 
}

std::vector<std::string> split_path(const std::string& path) {
  std::vector<std::string> result; 
  std::stringstream stream(path); 
  std::string item;
  while (std::getline(stream, item, '/')) {
    if (!item.empty()) result.push_back(item);
  }
  return result;
}

std::size_t content_length(const std::map<std::string, std::string>& headers) {
  auto it = headers.find("content-length");
  if (it == headers.end()) return 0;
  return static_cast<std::size_t>(std::stoull(it->second));
}

// Formats and sends an HTTP response[cite: 2]
void send_response(int descriptor, int status, const std::string& body = "", const std::string& type = "application/json") {
  const char* phrase = status == 200 ? "OK" : 
                       status == 201 ? "Created" : 
                       status == 204 ? "No Content" : 
                       status == 404 ? "Not Found" : 
                       status == 422 ? "Unprocessable Entity" : "Bad Request";
                       
  std::ostringstream response;
  response << "HTTP/1.1 " << status << " " << phrase << "\r\n"
           << "Content-Type: " << type << "\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n\r\n" 
           << body;
           
  send_all(descriptor, response.str());
}

std::string error_json(const std::string& message) {
  json_object* value = json_object_new_object(); 
  json_object_object_add(value, "error", json_object_new_string(message.c_str()));
  const std::string output = json_object_to_json_string_ext(value, JSON_C_TO_STRING_PLAIN); 
  json_object_put(value); 
  return output;
}

// Handles incoming client HTTP requests on the server[cite: 2]
void server_client(int descriptor, const Repository& repository) {
  try {
    std::string raw; 
    char buffer[64 * 1024];
    
    // Read headers[cite: 2]
    while (raw.find("\r\n\r\n") == std::string::npos) {
      const auto count = recv(descriptor, buffer, sizeof(buffer), 0);
      if (count <= 0) throw std::runtime_error("connection ended before headers");
      raw.append(buffer, static_cast<std::size_t>(count));
      if (raw.size() > 64 * 1024) throw std::runtime_error("headers too large");
    }
    
    const auto header_end = raw.find("\r\n\r\n");
    std::istringstream input(raw.substr(0, header_end));
    std::string request_line; 
    std::getline(input, request_line);
    
    std::istringstream request_stream(request_line); 
    std::string method, path, version; 
    request_stream >> method >> path >> version;
    
    // Parse headers into a map[cite: 2]
    std::map<std::string, std::string> headers; 
    std::string line;
    while (std::getline(input, line) && line != "\r") {
      const auto colon = line.find(':');
      if (colon != std::string::npos) { 
        auto key = line.substr(0, colon); 
        std::transform(key.begin(), key.end(), key.begin(), ::tolower); 
        
        auto value = line.substr(colon + 1); 
        while (!value.empty() && (value.front() == ' ' || value.back() == '\r')) { 
          if (value.front() == ' ') value.erase(0, 1); 
          if (!value.empty() && value.back() == '\r') value.pop_back(); 
        } 
        headers[key] = value; 
      }
    }
    
    const auto length = content_length(headers);
    std::string body = raw.substr(header_end + 4);
    
    // Read the remaining body bytes based on Content-Length[cite: 2]
    while (body.size() < length) { 
      const auto count = recv(descriptor, buffer, std::min(sizeof(buffer), length - body.size()), 0); 
      if (count <= 0) throw std::runtime_error("partial upload rejected"); 
      body.append(buffer, static_cast<std::size_t>(count)); 
    }
    if (body.size() > length) {
      body.resize(length);
    }
    
    const auto parts = split_path(path);
    
    // Route: GET /v1/info[cite: 2]
    if (method == "GET" && parts == std::vector<std::string>{"v1", "info"}) { 
      send_response(descriptor, 200, "{\"hash_algorithm\":\"sha256\"}"); 
    }
    // Route: POST /v1/objects/exists (Check if objects exist on peer)[cite: 2]
    else if (method == "POST" && parts == std::vector<std::string>{"v1", "objects", "exists"}) {
      json_object* input_json = json_tokener_parse(body.c_str()); 
      if (!input_json) throw std::runtime_error("bad JSON");
      
      json_object* objects = nullptr; 
      if (!json_object_object_get_ex(input_json, "objects", &objects) || !json_object_is_type(objects, json_type_array)) {
        throw std::runtime_error("objects list required");
      }
      
      json_object* output = json_object_new_object(); 
      json_object* present = json_object_new_array(); 
      json_object* missing = json_object_new_array();
      
      for (int i = 0; i < static_cast<int>(json_object_array_length(objects)); ++i) { 
        auto* entry = json_object_array_get_idx(objects, i); 
        json_object *kind = nullptr, *hash = nullptr; 
        
        if (!json_object_object_get_ex(entry, "kind", &kind) || !json_object_object_get_ex(entry, "hash", &hash)) {
          throw std::runtime_error("bad object id"); 
        }
        
        ObjectId id{parse_kind(json_object_get_string(kind)), json_object_get_string(hash)}; 
        auto* copy = json_object_new_object(); 
        json_object_object_add(copy, "kind", json_object_new_string(kind_name(id.kind).c_str())); 
        json_object_object_add(copy, "hash", json_object_new_string(id.hash.c_str())); 
        
        json_object_array_add(repository.has_object(id) ? present : missing, copy); 
      }
      
      json_object_object_add(output, "present", present); 
      json_object_object_add(output, "missing", missing); 
      
      const std::string answer = json_object_to_json_string_ext(output, JSON_C_TO_STRING_PLAIN); 
      json_object_put(output); 
      json_object_put(input_json); 
      send_response(descriptor, 200, answer);
    } 
    // Route: GET /v1/objects/<kind>/<hash> (Download object)[cite: 2]
    else if (method == "GET" && parts.size() == 4 && parts[0] == "v1" && parts[1] == "objects") { 
      send_response(descriptor, 200, repository.read_object({parse_kind(parts[2]), parts[3]}), "application/octet-stream"); 
    }
    // Route: PUT /v1/objects/<kind>/<hash> (Upload object)[cite: 2]
    else if (method == "PUT" && parts.size() == 4 && parts[0] == "v1" && parts[1] == "objects") { 
      ObjectId id{parse_kind(parts[2]), parts[3]}; 
      const bool created = repository.write_verified_object(id, body); 
      std::cout << "[SynapseFS] " << (created ? "received " : "already had ") 
                << kind_name(id.kind) << " " << id.hash.substr(0, 12) << "... (" << body.size() << " bytes)\n"; 
      send_response(descriptor, 201); 
    }
    // Route: Refs management (Branch heads)[cite: 2]
    else if (parts.size() >= 4 && parts[0] == "v1" && parts[1] == "refs" && parts[2] == "heads") {
      std::string branch; 
      for (std::size_t i = 3; i < parts.size(); ++i) {
        branch += (i == 3 ? "" : "/") + parts[i];
      }
      
      if (method == "GET") {
        send_response(descriptor, 200, "{\"head\":\"" + repository.read_ref(branch) + "\"}");
      } else if (method == "PUT") { 
        json_object* input_json = json_tokener_parse(body.c_str()); 
        json_object* head = nullptr; 
        
        if (!input_json || !json_object_object_get_ex(input_json, "head", &head)) {
          throw std::runtime_error("head required"); 
        }
        
        repository.update_ref(branch, json_object_get_string(head)); 
        if (input_json) json_object_put(input_json); 
        
        std::cout << "\n=== SynapseFS receive complete ===\nRepository: " << repository.root() 
                  << "\nBranch: " << branch << "\nPublished: " 
                  << repository.read_ref(branch).substr(0,12) << "...\n\n"; 
        send_response(descriptor, 204); 
      } else {
        throw std::runtime_error("unsupported ref method");
      }
    } else {
      send_response(descriptor, 404, error_json("unknown endpoint"));
    }
  } catch (const std::exception& error) { 
    std::cerr << "[SynapseFS] rejected request: " << error.what() << "\n"; 
    send_response(descriptor, 422, error_json(error.what())); 
  }
  close(descriptor);
}

// Queries a peer to find out which objects it already has[cite: 2]
std::set<ObjectId> peer_present(const Url& url, const std::set<ObjectId>& objects) {
  json_object* input = json_object_new_object(); 
  json_object* list = json_object_new_array();
  
  for (const auto& id : objects) { 
    auto* entry = json_object_new_object(); 
    json_object_object_add(entry, "kind", json_object_new_string(kind_name(id.kind).c_str())); 
    json_object_object_add(entry, "hash", json_object_new_string(id.hash.c_str())); 
    json_object_array_add(list, entry); 
  }
  
  json_object_object_add(input, "objects", list); 
  const std::string body = json_object_to_json_string_ext(input, JSON_C_TO_STRING_PLAIN); 
  json_object_put(input);
  
  const auto response = client_request(url, "POST", "/v1/objects/exists", body); 
  if (response.status != 200) {
    throw std::runtime_error("peer exists request failed");
  }
  
  json_object* value = json_tokener_parse(response.body.c_str()); 
  json_object* present = nullptr; 
  if (!value || !json_object_object_get_ex(value, "present", &present)) {
    throw std::runtime_error("invalid peer exists response"); 
  }
  
  std::set<ObjectId> result;
  for (int i = 0; i < static_cast<int>(json_object_array_length(present)); ++i) { 
    auto* entry = json_object_array_get_idx(present, i); 
    json_object *kind = nullptr, *hash = nullptr; 
    json_object_object_get_ex(entry, "kind", &kind); 
    json_object_object_get_ex(entry, "hash", &hash); 
    result.insert({parse_kind(json_object_get_string(kind)), json_object_get_string(hash)}); 
  }
  
  json_object_put(value); 
  return result;
}

int rank(ObjectKind kind) { 
  return kind == ObjectKind::block ? 0 : 
         kind == ObjectKind::artifact ? 1 : 
         kind == ObjectKind::tree ? 2 : 3; 
}

}  // namespace

PeerServer::PeerServer(Repository repository, std::uint16_t port) 
    : repository_(std::move(repository)), port_(port) {}

// Starts the TCP server loop[cite: 2]
void PeerServer::serve_forever() const {
  repository_.ensure_layout(); 
  const int listener = socket(AF_INET, SOCK_STREAM, 0); 
  if (listener < 0) throw std::runtime_error("cannot create server socket"); 
  
  int yes = 1; 
  setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)); 
  
  sockaddr_in address{}; 
  address.sin_family = AF_INET; 
  address.sin_addr.s_addr = INADDR_ANY; 
  address.sin_port = htons(port_); 
  
  if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || listen(listener, 32) != 0) { 
    close(listener); 
    throw std::runtime_error("cannot listen on requested port"); 
  } 
  
  std::cout << "\n=== SynapseFS peer ready ===\nRepository: " << repository_.root() 
            << "\nListening:  0.0.0.0:" << port_ 
            << "\nWaiting for a sender. Press Ctrl+C to stop.\n\n"; 
            
  while (true) { 
    const int client = accept(listener, nullptr, nullptr); 
    if (client >= 0) {
      // Spawn a detached thread for each incoming connection[cite: 2]
      std::thread(server_client, client, std::cref(repository_)).detach(); 
    }
  }
}

// Pushes local branch data to a remote peer[cite: 2]
SyncReport push_branch(const Repository& repository, const std::string& remote_url, const std::string& branch) {
  const Url url = parse_url(remote_url); 
  const auto head = repository.read_ref(branch); 
  const auto required = repository.reachable_objects(head); 
  
  // Calculate the difference between local and remote objects[cite: 2]
  const auto present = peer_present(url, required); 
  std::vector<ObjectId> missing; 
  
  for (const auto& id : required) {
    if (!present.count(id)) missing.push_back(id); 
  }
  
  // Send data sequentially based on object type rank[cite: 2]
  std::sort(missing.begin(), missing.end(), [](const auto& a, const auto& b) { 
    return std::make_tuple(rank(a.kind), a.hash) < std::make_tuple(rank(b.kind), b.hash); 
  }); 
  
  std::uint64_t bytes = 0; 
  std::cout << "[SynapseFS] compared " << required.size() << " objects: " 
            << present.size() << " already present, " << missing.size() << " to transfer\n"; 
            
  for (const auto& id : missing) { 
    auto data = repository.read_object(id); 
    std::cout << "[SynapseFS] sending " << kind_name(id.kind) << " " 
              << id.hash.substr(0,12) << "... (" << data.size() << " bytes)\n"; 
              
    if (client_request(url, "PUT", object_path(id), data, "application/octet-stream").status != 201) {
      throw std::runtime_error("peer rejected object"); 
    }
    bytes += data.size(); 
  } 
  
  if (client_request(url, "PUT", "/v1/refs/heads/" + branch, "{\"head\":\"" + head + "\"}").status != 204) {
    throw std::runtime_error("peer rejected ref update"); 
  }
  
  std::cout << "\n=== SynapseFS send complete ===\nBranch: " << branch 
            << "\nDestination: " << remote_url << "\nSent: " << missing.size() 
            << " new object(s), " << bytes << " bytes\nSkipped: " << present.size() << " existing object(s)\n\n"; 
            
  return {branch, head, required.size(), present.size(), missing.size(), bytes};
}

// Pulls branch data from a remote peer to local storage[cite: 2]
SyncReport pull_branch(const Repository& repository, const std::string& remote_url, const std::string& branch) {
  const Url url = parse_url(remote_url); 
  
  // Fetch remote branch head[cite: 2]
  const auto ref = client_request(url, "GET", "/v1/refs/heads/" + branch); 
  if (ref.status != 200) throw std::runtime_error("remote branch not found"); 
  
  json_object* value = json_tokener_parse(ref.body.c_str()); 
  json_object* head_value = nullptr; 
  if (!value || !json_object_object_get_ex(value, "head", &head_value)) {
    throw std::runtime_error("invalid remote ref"); 
  }
  
  const std::string head = json_object_get_string(head_value); 
  json_object_put(value); 
  
  std::vector<ObjectId> pending{{ObjectKind::commit, head}}; 
  std::set<ObjectId> visited; 
  std::uint64_t bytes = 0; 
  std::size_t present = 0, transferred = 0; 
  
  // Traverse and download missing objects iteratively[cite: 2]
  while (!pending.empty()) { 
    auto id = pending.back(); 
    pending.pop_back(); 
    if (visited.count(id)) continue; 
    
    if (repository.has_object(id)) { 
      repository.read_object(id); 
      ++present; 
    } else { 
      auto response = client_request(url, "GET", object_path(id)); 
      if (response.status != 200) throw std::runtime_error("remote object missing"); 
      repository.write_verified_object(id, response.body); 
      ++transferred; 
      bytes += response.body.size(); 
    } 
    
    visited.insert(id); 
    for (const auto& child : repository.direct_references(id)) {
      if (!visited.count(child)) pending.push_back(child); 
    }
  } 
  
  repository.update_ref(branch, head); 
  
  std::cout << "\n=== SynapseFS pull complete ===\nBranch: " << branch 
            << "\nReceived: " << transferred << " new object(s), " << bytes 
            << " bytes\nSkipped: " << present << " existing object(s)\n\n"; 
            
  return {branch, head, visited.size(), present, transferred, bytes};
}

}  // namespace synapsefs