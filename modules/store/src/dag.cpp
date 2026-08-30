#include <synapsefs/store/dag.hpp>

#include <algorithm>
#include <queue>

#include <synapsefs/store/manifest_store.hpp>

namespace sfs::store {

Status walk_commits(CommitStore& commits, std::span<const Oid> tips,
                    const std::function<bool(const Oid&, const format::Commit&)>& visit) {
    std::unordered_set<Oid> seen;
    // Max-heap on timestamp string (RFC3339 sorts lexicographically) so ties
    // and near-ties come out newest-first without a second pass.
    using Entry = std::pair<std::string, Oid>;
    std::priority_queue<Entry> pq;

    for (const auto& tip : tips) {
        if (!seen.insert(tip).second) continue;
        auto c = commits.read(tip);
        if (!c) return c.error();
        pq.emplace(c->timestamp, tip);
    }

    while (!pq.empty()) {
        auto [ts, oid] = pq.top();
        pq.pop();
        auto c = commits.get_cached(oid);
        if (!c) return c.error();

        bool keep_going = visit(oid, **c);
        if (!keep_going) continue;

        for (const auto& parent : (*c)->parents) {
            if (!seen.insert(parent).second) continue;
            auto pc = commits.read(parent);
            if (!pc) return pc.error();
            pq.emplace(pc->timestamp, parent);
        }
    }
    return {};
}

Result<bool> is_ancestor(CommitStore& commits, const Oid& maybe_ancestor, const Oid& of) {
    if (maybe_ancestor == of) return true;  // a commit is its own ancestor for this check

    std::unordered_set<Oid> visited;
    std::vector<Oid> stack{of};
    while (!stack.empty()) {
        Oid cur = stack.back();
        stack.pop_back();
        if (!visited.insert(cur).second) continue;
        if (cur == maybe_ancestor) return true;

        auto c = commits.read(cur);
        if (!c) return std::unexpected(c.error());
        for (const auto& p : c->parents) stack.push_back(p);
    }
    return false;
}

Result<std::optional<Oid>> merge_base(CommitStore& commits, const Oid& a, const Oid& b) {
    // Collect all ancestors of `a` (including a), then walk from `b` in
    // generation order until we hit the first one that's in that set.
    std::unordered_set<Oid> a_ancestors;
    {
        std::vector<Oid> stack{a};
        while (!stack.empty()) {
            Oid cur = stack.back();
            stack.pop_back();
            if (!a_ancestors.insert(cur).second) continue;
            auto c = commits.read(cur);
            if (!c) return std::unexpected(c.error());
            for (const auto& p : c->parents) stack.push_back(p);
        }
    }

    std::unordered_set<Oid> visited;
    std::queue<Oid> q;
    q.push(b);
    while (!q.empty()) {
        Oid cur = q.front();
        q.pop();
        if (!visited.insert(cur).second) continue;
        if (a_ancestors.count(cur)) return std::optional<Oid>(cur);

        auto c = commits.read(cur);
        if (!c) return std::unexpected(c.error());
        for (const auto& p : c->parents) q.push(p);
    }
    return std::optional<Oid>{};
}

Result<std::unordered_set<Oid>> reachable_objects(CommitStore& commits, ManifestStore& manifests,
                                                   std::span<const Oid> tips) {
    std::unordered_set<Oid> objects;

    std::optional<core::Error> err;
    auto st = walk_commits(commits, tips, [&](const Oid& commit_oid, const format::Commit& c) {
        objects.insert(commit_oid);
        objects.insert(c.manifest);
        objects.insert(c.topology);

        auto m = manifests.read(c.manifest);
        if (!m) {
            err = m.error();
            return false;
        }
        objects.insert(m->file.header_block);
        for (const auto& [_, g] : m->groups) {
            if (g.block) objects.insert(*g.block);
            if (g.diff_block) objects.insert(*g.diff_block);
        }
        return true;
    });
    if (!st) return std::unexpected(st.error());
    if (err) return std::unexpected(*err);
    return objects;
}

Result<std::vector<Oid>> have_probe(CommitStore& commits, std::span<const Oid> tips,
                                    std::size_t limit, std::uint32_t /*round*/) {
    // Exponentially widening strides back from each tip: generations
    // 1, 2, 4, 8, ... This is the classic git-style negotiation set, cheap to
    // compute and, for the common "a few commits apart" case, sufficient in
    // one round trip.
    std::vector<Oid> out;
    std::unordered_set<Oid> seen;

    for (const auto& tip : tips) {
        Oid cur = tip;
        std::uint32_t stride = 1;
        while (out.size() < limit) {
            if (seen.insert(cur).second) out.push_back(cur);
            std::uint32_t hops = stride;
            Oid next = cur;
            bool advanced = false;
            while (hops > 0) {
                auto c = commits.read(next);
                if (!c || c->parents.empty()) { advanced = false; break; }
                next = c->parents[0];
                advanced = true;
                --hops;
            }
            if (!advanced) break;
            cur = next;
            stride *= 2;
        }
    }
    return out;
}

}  // namespace sfs::store
