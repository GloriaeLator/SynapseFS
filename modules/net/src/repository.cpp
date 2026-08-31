#include <synapsefs/net/repository.hpp>

#include <openssl/sha.h>
#include <json-c/json.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fcntl.h>
#include <fstream>
#include <iomanip>s
#include <map>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace fs = std::filesystem;

namespace sfs::net {
namespace {

// Reads a file's complete contents into a string[cite: 3]
std::string read_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("missing file: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), {}};
}

// Synchronizes directory metadata to storage[cite: 3]
void fsync_directory(const fs::path& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY);
  if (descriptor >= 0) { 
    ::fsync(descriptor); 
    ::close(descriptor); 
  }
}

// Writes data to a temporary file and atomically renames it to ensure data integrity[cite: 3]
void atomic_write(const fs::path& destination, const std::string& bytes) {
  fs::create_directories(destination.parent_path());
  
  const auto temporary = destination.parent_path() / (".tmp-" + std::to_string(::getpid()) + "-" + std::to_string(std::rand()));
  const int descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  
  if (descriptor < 0) {
    throw std::runtime_error("cannot create temporary object");
  }
  
  try {
    std::size_t written = 0;
    while (written < bytes.size()) {
      const auto result = ::write(descriptor, bytes.data() + written, bytes.size() - written);
      if (result <= 0) throw std::runtime_error("cannot write temporary object");
      written += static_cast<std::size_t>(result);
    }
    
    if (::fsync(descriptor) != 0) {
      throw std::runtime_error("cannot fsync temporary object");
    }
    ::close(descriptor);
    
    // Commit the temporary file by renaming[cite: 3]
    fs::rename(temporary, destination);
    fsync_directory(destination.parent_path());
    
  } catch (...) {
    ::close(descriptor);
    std::error_code ignored;
    fs::remove(temporary, ignored); // Clean up the temp file on error[cite: 3]
    throw;
  }
}

std::string json_string(json_object* value) {
  if (!json_object_is_type(value, json_type_string)) {
    throw std::runtime_error("expected JSON string");
  }
  return json_object_get_string(value);
}

// Normalizes JSON representation to guarantee stable cryptographic hashing[cite: 3]
std::string canonical_value(json_object* value) {
  const auto type = json_object_get_type(value);
  
  if (type == json_type_object) {
    std::map<std::string, json_object*> ordered;
    json_object_object_foreach(value, key, child) { 
      ordered.emplace(key, child); 
    }
    
    std::string result = "{";
    bool first = true;
    for (const auto& [key, child] : ordered) {
      if (!first) result += ",";
      first = false;
      
      json_object* key_object = json_object_new_string(key.c_str());
      result += json_object_to_json_string_ext(key_object, JSON_C_TO_STRING_PLAIN);
      json_object_put(key_object);
      
      result += ":" + canonical_value(child);
    }
    return result + "}";
  }
  
  if (type == json_type_array) {
    std::string result = "[";
    for (int i = 0; i < static_cast <int>(json_object_array_length(value)); ++i) {
      if (i) result += ",";
      result += canonical_value(json_object_array_get_idx(value, i));
    }
    return result + "]";
  }
  
  return json_object_to_json_string_ext(value, JSON_C_TO_STRING_PLAIN);
}

void add_reference(std::set<ObjectId>& output, ObjectKind kind, const std::string& hash) {
  output.insert({kind, hash});
}

// Extracts all child ObjectIds referenced within a JSON object[cite: 3]
std::set<ObjectId> direct_references(const Repository& repository, const ObjectId& object) {
  if (object.kind == ObjectKind::block) return {}; // Blocks are raw data, no children[cite: 3]
  
  json_object* value = json_tokener_parse(repository.read_object(object).c_str());
  if (!value || !json_object_is_type(value, json_type_object)) {
    throw std::runtime_error("object is not a JSON object");
  }
  
  std::set<ObjectId> references;
  try {
    json_object* field = nullptr;
    
    if (object.kind == ObjectKind::commit) {
      if (!json_object_object_get_ex(value, "tree", &field)) {
        throw std::runtime_error("commit lacks tree");
      }
      add_reference(references, ObjectKind::tree, json_string(field));
      
      if (json_object_object_get_ex(value, "parents", &field)) {
        if (!json_object_is_type(field, json_type_array)) throw std::runtime_error("commit parents must be array");
        for (int i = 0; i < static_cast<int> (json_object_array_length(field)); ++i) {
          add_reference(references, ObjectKind::commit, json_string(json_object_array_get_idx(field, i)));
        }
      }
    } 
    else if (object.kind == ObjectKind::tree) {
      if (json_object_object_get_ex(value, "objects", &field)) {
        if (!json_object_is_type(field, json_type_array)) throw std::runtime_error("tree objects must be array");
        
        for (int i = 0; i < static_cast<int>(json_object_array_length(field)); ++i) {
          auto* entry = json_object_array_get_idx(field, i);
          json_object *kind = nullptr, *hash = nullptr;
          
          if (!json_object_object_get_ex(entry, "kind", &kind) || !json_object_object_get_ex(entry, "hash", &hash)) {
            throw std::runtime_error("bad tree entry");
          }
          add_reference(references, parse_kind(json_string(kind)), json_string(hash));
        }
      }
    } 
    else {  // artifact[cite: 3]
      if (json_object_object_get_ex(value, "payload_block", &field)) {
        add_reference(references, ObjectKind::block, json_string(field));
      }
      if (json_object_object_get_ex(value, "payload_blocks", &field)) {
        if (!json_object_is_type(field, json_type_array)) throw std::runtime_error("payload_blocks must be array");
        for (int i = 0; i < static_cast<int>(json_object_array_length(field)); ++i) {
          add_reference(references, ObjectKind::block, json_string(json_object_array_get_idx(field, i)));
        }
      }
    }
  } catch (...) { 
    json_object_put(value); 
    throw; 
  }
  
  json_object_put(value);
  return references;
}

}  // namespace

bool ObjectId::operator<(const ObjectId& other) const { 
  return std::tie(kind, hash) < std::tie(other.kind, other.hash); 
}

ObjectKind parse_kind(const std::string& value) {
  if (value == "block") return ObjectKind::block;
  if (value == "artifact") return ObjectKind::artifact;
  if (value == "tree") return ObjectKind::tree;
  if (value == "commit") return ObjectKind::commit;
  throw std::runtime_error("unknown object kind: " + value);
}

std::string kind_name(ObjectKind kind) {
  switch (kind) { 
    case ObjectKind::block: return "block"; 
    case ObjectKind::artifact: return "artifact"; 
    case ObjectKind::tree: return "tree"; 
    case ObjectKind::commit: return "commit"; 
  }
  throw std::runtime_error("invalid object kind");
}

void validate_hash(const std::string& value) {
  if (value.size() != 64 || !std::all_of(value.begin(), value.end(), [](unsigned char c) { 
    return std::isxdigit(c) && !std::isupper(c); 
  })) {
    throw std::runtime_error("invalid SHA-256 hash");
  }
}

void validate_branch(const std::string& value) { 
  if (value.empty() || value[0] == '/' || value.find("..") != std::string::npos) {
    throw std::runtime_error("invalid branch name"); 
  }
}

std::string canonical_json(const std::string& json_text) {
  json_object* value = json_tokener_parse(json_text.c_str());
  if (!value) {
    throw std::runtime_error("invalid JSON");
  }
  const auto result = canonical_value(value);
  json_object_put(value);
  return result;
}

// Repository Class Implementation[cite: 3]
Repository::Repository(fs::path root) : root_(std::move(root)) {}

void Repository::ensure_layout() const { 
  for (auto name : {"blocks", "artifacts", "trees", "commits"}) {
    fs::create_directories(root_ / ".synapsefs" / "objects" / name); 
  }
  fs::create_directories(root_ / ".synapsefs" / "refs" / "heads"); 
}

std::string Repository::digest(const std::string& bytes) const { 
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{}; 
  SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), digest.data()); 
  
  std::ostringstream output; 
  for (auto byte : digest) {
    output << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte); 
  }
  return output.str(); 
}

fs::path Repository::object_path(const ObjectId& id) const { 
  validate_hash(id.hash); 
  return root_ / ".synapsefs" / "objects" / (kind_name(id.kind) + "s") / id.hash; 
}

fs::path Repository::ref_path(const std::string& branch) const { 
  validate_branch(branch); 
  return root_ / ".synapsefs" / "refs" / "heads" / branch; 
}

bool Repository::has_object(const ObjectId& id) const { 
  return fs::is_regular_file(object_path(id)); 
}

// Validates hash on read[cite: 3]
std::string Repository::read_object(const ObjectId& id) const { 
  const auto bytes = read_file(object_path(id)); 
  if (digest(bytes) != id.hash) {
    throw std::runtime_error("hash mismatch for " + kind_name(id.kind) + " " + id.hash); 
  }
  return bytes; 
}

// Validates hash before write[cite: 3]
bool Repository::write_verified_object(const ObjectId& id, const std::string& bytes) const { 
  if (digest(bytes) != id.hash) throw std::runtime_error("received object hash mismatch"); 
  
  if (has_object(id)) { 
    read_object(id); 
    return false; // Not a new object[cite: 3]
  } 
  
  atomic_write(object_path(id), bytes); 
  return true; 
}

std::string Repository::read_ref(const std::string& branch) const { 
  auto value = read_file(ref_path(branch)); 
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
    value.pop_back(); 
  }
  validate_hash(value); 
  return value; 
}

void Repository::update_ref(const std::string& branch, const std::string& commit_hash) const { 
  validate_hash(commit_hash); 
  read_object({ObjectKind::commit, commit_hash}); // Ensure commit exists locally[cite: 3]
  atomic_write(ref_path(branch), commit_hash + "\n"); 
}

std::set<ObjectId> Repository::direct_references(const ObjectId& object) const { 
  return ::sfs::net::direct_references(*this, object); 
}

// Computes the closure of all objects reachable from a head commit[cite: 3]
std::set<ObjectId> Repository::reachable_objects(const std::string& head_commit) const { 
  std::vector<ObjectId> pending{{ObjectKind::commit, head_commit}}; 
  std::set<ObjectId> visited; 
  
  while (!pending.empty()) { 
    auto current = pending.back(); 
    pending.pop_back(); 
    
    if (visited.count(current)) continue; 
    
    read_object(current); 
    visited.insert(current); 
    
    for (const auto& child : direct_references(current)) {
      if (!visited.count(child)) pending.push_back(child); 
    }
  } 
  return visited; 
}

}  // namespace synapsefs