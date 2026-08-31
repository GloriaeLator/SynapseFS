#pragma once

#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace sfs::net {

enum class ObjectKind { block, artifact, tree, commit };

struct ObjectId {
  ObjectKind kind;
  std::string hash;
  bool operator<(const ObjectId& other) const;
};

struct SyncReport {
  std::string branch;
  std::string head;
  std::size_t objects_total{};
  std::size_t objects_already_present{};
  std::size_t objects_transferred{};
  std::uint64_t bytes_transferred{};
}; 

class Repository {
 public:
  explicit Repository(std::filesystem::path root);

  void ensure_layout() const;
  std::string digest(const std::string& bytes) const;
  std::filesystem::path object_path(const ObjectId& id) const;
  std::filesystem::path ref_path(const std::string& branch) const;
  bool has_object(const ObjectId& id) const;
  std::string read_object(const ObjectId& id) const;
  bool write_verified_object(const ObjectId& id, const std::string& bytes) const;
  std::string read_ref(const std::string& branch) const;
  void update_ref(const std::string& branch, const std::string& commit_hash) const;
  std::set<ObjectId> direct_references(const ObjectId& object) const;
  std::set<ObjectId> reachable_objects(const std::string& head_commit) const;
  const std::filesystem::path& root() const { return root_; }

 private:
  std::filesystem::path root_;
};

ObjectKind parse_kind(const std::string& value);
std::string kind_name(ObjectKind kind);
void validate_hash(const std::string& value);
void validate_branch(const std::string& value);
std::string canonical_json(const std::string& json_text);

}  // namespace synapsefs
