#include "server-fs.h"

#include "server-common.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <unordered_set>

#if defined(_WIN32)
#  include <sys/stat.h>
#elif defined(__APPLE__)
#  include <sys/stat.h>
#elif defined(__linux__)
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#  include <linux/stat.h>
#endif

namespace fs = std::filesystem;

namespace server_fs {

// Returns the user's home directory from $HOME (Unix) or %USERPROFILE% (Windows).
// Empty string if neither is set.
static std::string home_dir() {
    const char * h = std::getenv("HOME");
    if (h && *h) return std::string(h);
    const char * u = std::getenv("USERPROFILE");
    if (u && *u) return std::string(u);
    return std::string();
}

std::vector<std::string> effective_roots(
        const std::vector<std::string> & configured,
        std::string & err) {
    std::vector<std::string> result;

    if (configured.empty()) {
        const std::string h = home_dir();
        if (h.empty()) {
            err = "no --browse-root configured and $HOME is not set";
            return {};
        }
        std::error_code ec;
        fs::path canon = fs::weakly_canonical(h, ec);
        if (ec) {
            err = "failed to canonicalize $HOME (" + h + "): " + ec.message();
            return {};
        }
        if (!fs::is_directory(canon, ec)) {
            err = "$HOME is not a directory: " + canon.string();
            return {};
        }
        result.push_back(canon.string());
        return result;
    }

    for (const auto & raw : configured) {
        if (raw.empty()) continue;
        std::error_code ec;
        fs::path canon = fs::weakly_canonical(raw, ec);
        if (ec || !fs::is_directory(canon, ec)) {
            // skip invalid root - error reporting happens below if no root
            // survives.
            continue;
        }
        result.push_back(canon.string());
    }

    if (result.empty()) {
        err = "no valid --browse-root directories to search";
        return {};
    }
    return result;
}

static bool is_child_of(const std::string & path, const std::string & root) {
    // strict prefix with separator boundary, OR exact match
    if (path == root) return true;
    if (path.size() <= root.size()) return false;
    if (path.compare(0, root.size(), root) != 0) return false;
    const char c = path[root.size()];
    return c == '/' || c == '\\';
}

std::string resolve_path(
        const std::string & path,
        const std::vector<std::string> & allowed_roots,
        std::string & err) {
    if (allowed_roots.empty()) {
        err = "filesystem browsing is not enabled (no roots available)";
        return {};
    }

    std::error_code ec;
    fs::path raw;

    if (path.empty()) {
        raw = fs::path(allowed_roots[0]);
    } else {
        raw = fs::path(path);
        if (!raw.is_absolute()) {
            // relative path: resolve against the first allowed root so the
            // walk stays inside the user's configured scope.
            raw = fs::path(allowed_roots[0]) / raw;
        }
    }

    fs::path canon = fs::weakly_canonical(raw, ec);
    if (ec) {
        err = "failed to resolve path: " + path + " (" + ec.message() + ")";
        return {};
    }

    const std::string canon_str = canon.string();

    bool inside = false;
    for (const auto & root : allowed_roots) {
        if (is_child_of(canon_str, root)) {
            inside = true;
            break;
        }
    }
    if (!inside) {
        err = "path is outside the configured --browse-root(s): " + canon_str;
        return {};
    }

    if (!fs::exists(canon)) {
        err = "path does not exist: " + canon_str;
        return {};
    }

    return canon_str;
}

static const std::unordered_set<std::string> & junk_dir_names() {
    static const std::unordered_set<std::string> names = {
        ".git", ".svn", ".hg", "node_modules", "__pycache__",
        ".venv", "venv", "dist", "build", "target", ".cache", ".idea", ".vscode",
        // system-level folders under $HOME that mostly contain caches / app data
        // (e.g. ~/Library/Application Support, ~/Library/Caches, ~/Library/Logs,
        // ~/Caches). Their contents rank above real working dirs in naive
        // recency-based sorting, so we skip them during walk entirely.
        "Library", "Caches",
    };
    return names;
}

static int64_t to_unix_seconds(const fs::file_time_type & ft) {
    // file_time_type's epoch differs by platform and C++17 has no portable way to convert
    // it to time_t. Use a clock-delta approach: measure how far `ft` is from `file_clock::now()`
    // and apply the same delta to `system_clock::now()`.
    const auto file_now = fs::file_time_type::clock::now();
    const auto sys_now  = std::chrono::system_clock::now();
    const auto delta    = ft - file_now;
    const auto sys      = std::chrono::time_point_cast<std::chrono::seconds>(sys_now + delta);
    return sys.time_since_epoch().count();
}

// Entry is "hidden" if any segment along its path starts with a dot. The leaf
// basename is covered as a special case, so a folder literally named '.foo'
// inside ~/git stays hidden. Mirrors Unix shell / GUI file manager conventions
// of ignoring .git, .config, etc.
static bool is_hidden_path(const fs::path & p) {
    for (const auto & seg : p) {
        const std::string name = seg.string();
        if (name.empty() || name == "." || name == "..") continue;
        if (name[0] == '.') return true;
    }
    return false;
}

// Best-effort creation / birth time, in unix seconds. Falls back to mtime when
// the platform/filesystem does not expose a creation timestamp (older Linux
// kernels, filesystems without birthtime support, etc.).
static int64_t get_added_time(const fs::path & path, int64_t mtime) {
#if defined(_WIN32)
    // windows: _wstat64 exposes creation time as st_ctime
    struct _stat64 st;
    if (_wstat64(path.c_str(), &st) == 0 && st.st_ctime > 0) {
        return (int64_t) st.st_ctime;
    }
    return mtime;
#elif defined(__APPLE__)
    // macOS: st_birthtime is available since 10.12
    struct stat st;
    if (stat(path.string().c_str(), &st) == 0 && st.st_birthtime > 0) {
        return (int64_t) st.st_birthtime;
    }
    return mtime;
#elif defined(__linux__)
    // Linux: use statx() to ask for STATX_BTIME; many filesystems (ext4 with
    // recent kernels, btrfs, xfs, zfs) honour it.
    struct statx stx;
    if (statx(AT_FDCWD, path.string().c_str(),
              AT_SYMLINK_NOFOLLOW | AT_NO_AUTOMOUNT,
              STATX_BTIME, &stx) == 0 &&
        (stx.stx_mask & STATX_BTIME) != 0 &&
        stx.stx_btime.tv_sec > 0) {
        return (int64_t) stx.stx_btime.tv_sec;
    }
    // POSIX fallback: ctime (inode change), at least distinct from mtime.
    struct stat st;
    if (stat(path.string().c_str(), &st) == 0 && st.st_ctime > 0) {
        return (int64_t) st.st_ctime;
    }
    return mtime;
#else
    (void) path;
    return mtime;
#endif
}

static bool contains_ci(const std::string & haystack, const std::string & needle) {
    if (needle.empty()) return true;
    if (needle.size() > haystack.size()) return false;
    for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        bool ok = true;
        for (size_t j = 0; j < needle.size(); ++j) {
            if (std::tolower((unsigned char) haystack[i + j]) !=
                std::tolower((unsigned char) needle[j])) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

static bool starts_with_ci(const std::string & s, const std::string & prefix) {
    if (prefix.size() > s.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower((unsigned char) s[i]) !=
            std::tolower((unsigned char) prefix[i])) {
            return false;
        }
    }
    return true;
}

namespace {

struct match_tier {
    int tier;             // 0 = exact, 1 = prefix, 2 = substring
    std::string lower;    // for alphabetical tiebreak
    bool hidden;          // any path segment starts with '.'
    int depth;            // number of segments below the search root
    int64_t modified;     // mtime, recency
    int64_t added;        // birth/creation time, recency
};

// Single-token (no slash) query: match against the entry's basename only.
// Returns 0 = exact, 1 = prefix, 2 = substring, 3 = no match.
static int classify_basename(const std::string & name, const std::string & query) {
    if (query.empty()) return 0;
    if (name.size() == query.size() && starts_with_ci(name, query)) return 0;
    if (starts_with_ci(name, query)) return 1;
    if (contains_ci(name, query)) return 2;
    return 3;
}

// Split a query on '/' and '\' (forward / back slash) and discard empty
// segments. "git/llama" -> ["git", "llama"]. Empty input -> empty vector.
static std::vector<std::string> split_query_on_slash(const std::string & query) {
    std::vector<std::string> result;
    std::string current;
    for (char c : query) {
        if (c == '/' || c == '\\') {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) result.push_back(current);
    return result;
}

static bool query_has_slash(const std::string & query) {
    for (char c : query) {
        if (c == '/' || c == '\\') return true;
    }
    return false;
}

// A query routes to path-like matching only when splitting it on '/' /\ '
// yields 2+ non-empty segments. "git" or "git/" stay on the basename
// track; "git/llama" goes to pathlike.
static bool query_is_pathlike(const std::string & query) {
    return split_query_on_slash(query).size() >= 2;
}

// Path-like query (contains a '/'): greedy left-to-right match of each
// query segment against successive path segments. Per-segment match kind
// (exact / prefix / substring) is tracked; final tier is the worst of all
// matched segments. Returns 3 if any query segment has no matching path
// segment.
static int classify_pathlike(const fs::path & entry_path, const std::vector<std::string> & q_segs) {
    if (q_segs.empty()) return 0;

    std::vector<std::string> p_segs;
    p_segs.reserve(8);
    for (const auto & seg : entry_path) {
        std::string s = seg.string();
        if (!s.empty()) p_segs.push_back(std::move(s));
    }

    int worst_tier = 0;
    size_t pi = 0;
    for (const auto & qs : q_segs) {
        bool found = false;
        for (; pi < p_segs.size(); ++pi) {
            const auto & ps = p_segs[pi];
            if (ps.size() == qs.size() && starts_with_ci(ps, qs)) {
                ++pi;
                found = true;
                break;
            }
            if (starts_with_ci(ps, qs)) {
                if (worst_tier < 1) worst_tier = 1;
                ++pi;
                found = true;
                break;
            }
            if (contains_ci(ps, qs)) {
                if (worst_tier < 2) worst_tier = 2;
                ++pi;
                found = true;
                break;
            }
        }
        if (!found) return 3;
    }
    return worst_tier;
}

// Count of non-trivial path segments between `root` and `p`. The leaf
// basename is included ("git/llama.cpp" -> 2, "git/llama.cpp/foo" -> 3).
// Used as a sort key so entries closer to the search root surface first -
// users typically pin their projects just below the browse root.
static int depth_below(const fs::path & p, const fs::path & root) {
    const fs::path rel = p.lexically_relative(root);
    int n = 0;
    for (const auto & seg : rel) {
        const std::string s = seg.string();
        if (s.empty() || s == "." || s == "..") continue;
        ++n;
    }
    return n;
}

} // namespace

bool search(
        const std::string & root,
        const search_options & opts,
        std::vector<search_entry> & results,
        std::string & err) {
    results.clear();

    std::error_code ec;
    fs::path root_path = root;
    if (!fs::is_directory(root_path, ec) || ec) {
        err = "not a directory: " + root;
        return false;
    }

    const std::string query_lower = [&]() {
        std::string s;
        s.reserve(opts.query.size());
        for (char c : opts.query) s.push_back(std::tolower((unsigned char) c));
        return s;
    }();

    // depth-limited iterative DFS, skipping junk dirs and any entry strictly
    // outside the depth budget.
    struct frame {
        fs::path dir;
        size_t depth;
    };
    std::vector<frame> stack;
    stack.push_back({root_path, 0});

    std::vector<std::pair<search_entry, match_tier>> scored;

    while (!stack.empty()) {
        auto frame = stack.back();
        stack.pop_back();

        fs::directory_iterator it(frame.dir, fs::directory_options::skip_permission_denied, ec);
        if (ec) continue;

        for (const auto & entry : it) {
            const std::string name = entry.path().filename().string();

            std::error_code is_ec;
            const bool is_dir = entry.is_directory(is_ec);
            const bool is_file = !is_dir && entry.is_regular_file(is_ec);

            if (is_dir) {
                if (junk_dir_names().count(name) > 0) continue;
                if (frame.depth + 1 < (size_t) opts.max_depth) {
                    stack.push_back({entry.path(), frame.depth + 1});
                }
            }

            // type filter
            if (opts.type == entry_type_filter::directory && !is_dir) continue;
            if (opts.type == entry_type_filter::file && !is_file) continue;

            // match filter
            int tier_int;
            const auto q_segs = split_query_on_slash(opts.query);
            if (q_segs.size() >= 2) {
                tier_int = classify_pathlike(entry.path(), q_segs);
            } else {
                const std::string effective = q_segs.empty() ? opts.query : q_segs[0];
                tier_int = classify_basename(name, effective);
            }
            if (!query_lower.empty()) {
                bool matched = false;
                switch (opts.match) {
                    case match_mode::prefix:
                        matched = tier_int <= 1;
                        break;
                    case match_mode::substring:
                    default:
                        matched = tier_int <= 2;
                        break;
                }
                if (!matched) continue;
            }

            // visibility filter: by default drop entries that live under a
            // "."-prefixed parent. The walk still recurses INTO those so
            // entries below them are findable when the user opts in.
            const bool hidden = is_hidden_path(entry.path());
            if (!opts.show_hidden && hidden) continue;

            search_entry e;
            e.name   = name;
            e.type   = is_dir ? "directory" : "file";
            e.parent = entry.path().parent_path().string();
            e.path   = entry.path().string();

            if (is_file) {
                std::error_code sz_ec;
                auto sz = entry.file_size(sz_ec);
                if (!sz_ec) e.size = (int64_t) sz;
            }

            std::error_code tm_ec;
            auto mtime = entry.last_write_time(tm_ec);
            if (!tm_ec) e.modified = to_unix_seconds(mtime);
            e.added = get_added_time(entry.path(), e.modified);

            match_tier mt;
            mt.tier     = tier_int;
            mt.lower    = name;
            mt.hidden   = hidden;
            mt.depth    = depth_below(entry.path(), root_path);
            mt.modified = e.modified;
            mt.added    = e.added;
            scored.emplace_back(std::move(e), mt);
        }
    }

    // rank cascade:
    //   1. tier (exact < prefix < substring) - textual relevance first
    //   2. !hidden                            - non-hidden entries first
    //   3. depth from search root ascending  - shallower paths (closer to
    //                                          the user's browse root) win over
    //                                          deeply nested matches
    //   4. -modified                         - most recently modified first
    //   5. -added                            - most recently created first
    //   6. alphabetical by raw name          - final tiebreaker
    std::sort(scored.begin(), scored.end(),
        [](const auto & a, const auto & b) {
            const auto & x = a.second;
            const auto & y = b.second;
            if (x.tier != y.tier) return x.tier < y.tier;
            if (x.hidden != y.hidden) return !x.hidden;
            if (x.depth != y.depth) return x.depth < y.depth;
            if (x.modified != y.modified) return x.modified > y.modified;
            if (x.added != y.added) return x.added > y.added;
            return x.lower < y.lower;
        });

    if ((int) scored.size() > opts.limit) scored.resize(opts.limit);

    results.reserve(scored.size());
    for (auto & p : scored) results.push_back(std::move(p.first));

    return true;
}

} // namespace server_fs

namespace server_fs {

// True when `path` is contained in `root` with a separator boundary
// (or is exactly equal to `root`). Mirrors the private is_child_of used
// by resolve_path so the git walk enforces the same scope.
static bool is_child_of_or_eq(const std::string & path, const std::string & root) {
    if (path == root) return true;
    if (path.size() <= root.size()) return false;
    if (path.compare(0, root.size(), root) != 0) return false;
    const char c = path[root.size()];
    return c == '/' || c == '\\';
}

// Trim trailing carriage returns / spaces so `.git/HEAD` lines from
// Windows checkouts don't leak '\r' into the branch name we return.
static void rstrip(std::string & s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
}

bool git_status(
        const std::string & path,
        const std::vector<std::string> & allowed_roots,
        git_info & info,
        std::string & err) {
    info = {};

    std::error_code ec;
    fs::path cur = fs::weakly_canonical(path, ec);
    if (ec) {
        err = "failed to resolve path: " + ec.message();
        return false;
    }

    // `.git` typically sits at the project root, a few levels above the
    // cwd. A bounded walk prevents us from accidentally probing every
    // ancestor up to the filesystem root in pathological cases.
    constexpr int MAX_DEPTH = 8;
    for (int depth = 0; depth <= MAX_DEPTH; ++depth) {
        const std::string cur_str = cur.string();

        // Confirm the current candidate is still inside an allowed root
        // before touching the filesystem. We bail out as soon as we cross
        // outside the configured browse scope - the equivalent of
        // resolve_path's containment check.
        bool inside = false;
        for (const auto & root : allowed_roots) {
            if (is_child_of_or_eq(cur_str, root)) {
                inside = true;
                break;
            }
        }
        if (!inside) {
            err = "no git repository found above " + path;
            return false;
        }

        const fs::path git_dir = cur / ".git";
        std::error_code is_ec;

        // Standard layout: `.git/` is a directory containing HEAD, refs/, etc.
        if (fs::is_directory(git_dir, is_ec)) {
            std::ifstream head(git_dir / "HEAD");
            if (head) {
                std::string line;
                if (std::getline(head, line)) {
                    rstrip(line);
                    const std::string ref_prefix = "ref: refs/heads/";
                    if (line.rfind(ref_prefix, 0) == 0) {
                        info.is_repo = true;
                        info.root = cur_str;
                        info.branch = line.substr(ref_prefix.size());
                        if (info.branch.empty()) info.branch = "detached";
                    } else if (line.size() == 40 && line.find(' ') == std::string::npos) {
                        // Detached HEAD: `.git/HEAD` holds a bare SHA rather
                        // than a refs pointer.
                        info.is_repo = true;
                        info.root = cur_str;
                        info.branch = "detached";
                        info.sha = line;
                    } else {
                        // Packed refs / partial clone / worktrees can yield
                        // other shapes; surface a soft "detached" so the UI
                        // still shows the repo without a misleading branch.
                        info.is_repo = true;
                        info.root = cur_str;
                        info.branch = "detached";
                    }
                    return true;
                }
            }
            // `.git` exists but HEAD is missing/unreadable - treat as a
            // bare-bones repo so the UI can surface the path even if it
            // can't name a branch.
            info.is_repo = true;
            info.root = cur_str;
            info.branch = "detached";
            return true;
        }

        // Submodule / gitfile layout: `.git` is a regular file whose
        // body is "gitdir: <relative path>". We don't chase the link
        // (it can reach outside the repo root) so we just mark the
        // current directory as a repo and return without a branch.
        std::error_code reg_ec;
        if (fs::is_regular_file(git_dir, reg_ec)) {
            info.is_repo = true;
            info.root = cur_str;
            info.branch = "submodule";
            return true;
        }

        const fs::path parent = cur.parent_path();
        if (parent == cur || parent.empty()) break;
        cur = parent;
    }

    err = "no .git found above " + path;
    return false;
}

} // namespace server_fs
