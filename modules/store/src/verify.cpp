#include <synapsefs/store/verify.hpp>

namespace sfs::store {

Status check_ancestor_invariant(CommitStore& commits, const Oid& commit_oid,
                                const format::Manifest& m) {
    for (const auto& [name, g] : m.groups) {
        if (g.mode != format::GroupMode::Delta) continue;
        if (!g.base) continue;  // caught by Manifest::validate() already

        auto anc = is_ancestor(commits, g.base->commit, commit_oid);
        if (!anc) return std::unexpected(anc.error());
        if (!*anc) {
            return SFS_ERR(AncestorInvariantViolated,
                           "delta group's base commit is not an ancestor of this commit",
                           name);
        }
    }
    return {};
}

Result<VerifyReport> verify(core::IBlockStore& blocks, CommitStore& commits,
                            ManifestStore& manifests, RefStore& refs, std::span<const Oid> tips,
                            const VerifyOptions& opts) {
    VerifyReport report;

    auto add_finding = [&](core::ErrKind kind, const Oid& obj, std::string detail,
                           std::optional<std::uint32_t> chunk = std::nullopt,
                           std::optional<std::string> group = std::nullopt) {
        report.findings.push_back(VerifyFinding{kind, obj, std::move(detail), chunk,
                                                 std::move(group)});
        report.objects_checked++;
    };

    // (0) opts.repair: journal recovery needs a store::Journal bound to
    // .synapsefs/journal, which this function is not given (verify() takes
    // stores and refs, not RepoPaths). By this module's own contract
    // (journal.hpp: "the caller must then refuse, not improvise"), the CLI's
    // `verify --repair` command constructs a Journal itself and calls
    // Journal::recover() BEFORE calling verify() here, so recovery has
    // already happened by the time this function runs. Nothing to do here.

    // (1) every ref resolves to a commit that exists.
    auto heads = refs.list_heads();
    if (!heads) {
        add_finding(heads.error().kind, Oid{}, heads.error().what);
        return report;
    }
    for (const auto& ref : *heads) {
        auto c = commits.read(ref.target);
        if (!c) {
            add_finding(core::ErrKind::RefNotFound, ref.target,
                       "ref " + ref.name + " points at a missing/invalid commit");
            if (opts.stop_on_first_error) return report;
        }
    }

    // (2)-(7): walk the DAG from `tips`, checking each commit, its manifest,
    // buffer layout, referenced blocks, and (for deltas) the ancestor
    // invariant + chain depth.
    std::unordered_set<Oid> seen_blocks;

    auto st = walk_commits(commits, tips, [&](const Oid& commit_oid, const format::Commit& c) {
        report.commits_walked++;

        // (2) commit hashes to its id — CommitStore::read() already checked
        // this on load and Commit::parse() already checked canonical
        // round-trip, so reaching here means both passed.

        // (3)+(4) manifest exists, well-formed, buffer layout valid —
        // Manifest::parse() already validated buffer layout on load.
        auto m = manifests.read(c.manifest);
        if (!m) {
            add_finding(m.error().kind, c.manifest, m.error().what);
            return !opts.stop_on_first_error;
        }

        // (5) every referenced block exists and matches its address.
        auto check_block = [&](const Oid& oid, core::ObjectKind kind, std::string_view what) {
            if (seen_blocks.count(oid)) return;
            seen_blocks.insert(oid);
            report.objects_checked++;

            if (opts.full) {
                if (auto vst = blocks.verify_block(oid, kind); !vst) {
                    add_finding(vst.error().kind, oid, std::string(what) + ": " + vst.error().what);
                }
            } else {
                auto exists = blocks.contains(oid);
                if (!exists || !*exists) {
                    add_finding(core::ErrKind::ObjectNotFound, oid,
                               std::string(what) + ": block missing");
                }
            }
        };

        check_block(m->file.header_block, core::ObjectKind::Header, "header block");
        for (const auto& [gname, g] : m->groups) {
            if (g.block) check_block(*g.block, core::ObjectKind::Raw, "group '" + gname + "'");
            if (g.diff_block)
                check_block(*g.diff_block, core::ObjectKind::Diff, "group '" + gname + "' diff");
        }

        // (6) ancestor invariant for every delta group.
        if (auto ai = check_ancestor_invariant(commits, commit_oid, *m); !ai) {
            add_finding(ai.error().kind, commit_oid, ai.error().what);
        }

        // (7) chain_depth consistency: a Delta group's chain_depth must be
        // exactly one more than its base group's chain_depth in the base
        // commit's manifest.
        for (const auto& [gname, g] : m->groups) {
            if (g.mode != format::GroupMode::Delta || !g.base) continue;
            auto base_manifest = manifests.manifest_for(g.base->commit);
            if (!base_manifest) {
                add_finding(base_manifest.error().kind, g.base->commit,
                           "group '" + gname + "': cannot load base commit's manifest");
                continue;
            }
            const auto* base_group = (*base_manifest)->find_group(g.base->group);
            if (!base_group) {
                add_finding(core::ErrKind::MalformedObject, g.base->commit,
                           "group '" + gname + "': base group '" + g.base->group +
                               "' not found in base manifest");
                continue;
            }
            if (g.chain_depth != base_group->chain_depth + 1) {
                add_finding(core::ErrKind::MalformedObject, commit_oid,
                           "group '" + gname + "': chain_depth " +
                               std::to_string(g.chain_depth) + " != base's " +
                               std::to_string(base_group->chain_depth) + " + 1");
            }
        }

        return !(opts.stop_on_first_error && !report.findings.empty());
    });

    if (!st) {
        add_finding(st.error().kind, Oid{}, st.error().what);
    }

    return report;
}

}  // namespace sfs::store
