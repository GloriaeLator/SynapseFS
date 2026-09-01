#ifndef SYNAPSE_SYNC_HPP    
#define SYNAPSE_SYNC_HPP

#include <string>

namespace sfs::net {

    // Expects remote_url in the format "IP:PORT" (e.g., "127.0.0.1:8080")
    bool push(const std::string& remote_url);
    
    // Expects remote_url in the format "IP:PORT"
    bool pull(const std::string& remote_url);
    
    // Starts a TCP server listening for push/pull requests
    void serve(int port);

}

#endif // SYNAPSE_SYNC_H