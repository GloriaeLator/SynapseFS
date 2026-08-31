#pragma once

#include <synapsefs/net/repository.hpp>

#include <cstdint>
#include <string>

namespace sfs::net {

class PeerServer {
 public:
  PeerServer(Repository repository, std::uint16_t port);
  void serve_forever() const;

 private:
  Repository repository_;
  std::uint16_t port_;
};

SyncReport push_branch(const Repository& repository, const std::string& remote_url, const std::string& branch);
SyncReport pull_branch(const Repository& repository, const std::string& remote_url, const std::string& branch);

}  // namespace synapsefs
