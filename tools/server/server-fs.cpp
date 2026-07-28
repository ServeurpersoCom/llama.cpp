#include "server-fs.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <system_error>

namespace fs = std::filesystem;

namespace server_fs {

std::string home_dir() {
    const char * h = std::getenv("HOME");
    if (h && *h) return std::string(h);
    const char * u = std::getenv("USERPROFILE");
    if (u && *u) return std::string(u);
    return std::string();
}

std::string expand_home(const std::string & path) {
    if (path.empty() || path[0] != '~') return path;
    if (path.size() > 1 && path[1] != '/' && path[1] != '\\') return path;
    const std::string home = home_dir();
    if (home.empty()) return path;
    return home + path.substr(1);
}

const std::unordered_set<std::string> & junk_dir_names() {
    static const std::unordered_set<std::string> names = {
        ".git", ".svn", ".hg", "node_modules", "__pycache__",
        ".venv", "venv", "dist", "build", "target", ".cache", ".idea", ".vscode",
    };
    return names;
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
            // skip invalid root - error reporting happens below if no root survives
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

// strict prefix with separator boundary, OR exact match
static bool is_child_of(const std::string & path, const std::string & root) {
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

    if (!fs::exists(canon, ec) || ec) {
        err = "path does not exist: " + canon_str;
        return {};
    }

    return canon_str;
}

static int64_t to_unix_seconds(const fs::file_time_type & ft) {
    // file_time_type's epoch differs by platform and C++17 has no portable way to convert
    // it to time_t; measure how far `ft` is from file_clock::now() and apply the same
    // delta to system_clock::now()
    const auto file_now = fs::file_time_type::clock::now();
    const auto sys_now  = std::chrono::system_clock::now();
    const auto delta    = ft - file_now;
    const auto sys      = std::chrono::time_point_cast<std::chrono::seconds>(sys_now + delta);
    return sys.time_since_epoch().count();
}

static std::string to_lower(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) out.push_back((char) std::tolower((unsigned char) c));
    return out;
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

// one walked entry with everything query-time ranking needs precomputed
struct walk_entry {
    std::string name;
    std::string name_lower;
    std::string path;                 // absolute
    std::string parent;               // absolute
    std::vector<std::string> segs;    // path segments below the walk root
    bool is_dir;
    bool hidden;                      // any segment below the root starts with '.'
    int depth;                        // number of segments below the root
    int64_t size;
    int64_t modified;
};

// Single-token (no slash) query: match against the entry's basename only.
// Returns 0 = exact, 1 = prefix, 2 = substring, 3 = no match.
int classify_basename(const std::string & name, const std::string & query) {
    if (name.size() == query.size() && starts_with_ci(name, query)) return 0;
    if (starts_with_ci(name, query)) return 1;
    if (contains_ci(name, query)) return 2;
    return 3;
}

// Split a query on '/' and '\', discarding empty segments: "git/llama" -> ["git", "llama"]
std::vector<std::string> split_query_on_slash(const std::string & query) {
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

// Path-like query (2+ segments): greedy left-to-right match of each query
// segment against successive path segments below the root. Final tier is the
// worst per-segment match; 3 if any query segment has no matching path segment.
int classify_pathlike(const std::vector<std::string> & p_segs, const std::vector<std::string> & q_segs) {
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

// Depth-limited iterative DFS from `root`, skipping junk dirs.
void walk_root(const fs::path & root, int max_depth, std::vector<walk_entry> & out) {
    out.clear();

    struct frame {
        fs::path dir;
        std::vector<std::string> segs; // dir's segments below the root
        bool hidden;                   // any segment in `segs` starts with '.'
    };
    std::vector<frame> stack;
    stack.push_back({root, {}, false});

    std::error_code ec;
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
            if (!is_dir && !is_file) continue; // skip sockets, broken symlinks, etc.
            if (is_dir && junk_dir_names().count(name) > 0) continue;

            const bool hidden = frame.hidden || (!name.empty() && name[0] == '.');

            walk_entry we;
            we.name       = name;
            we.name_lower = to_lower(name);
            we.path       = entry.path().string();
            we.parent     = frame.dir.string();
            we.segs       = frame.segs;
            we.segs.push_back(name);
            we.is_dir     = is_dir;
            we.hidden     = hidden;
            we.depth      = (int) we.segs.size();
            we.size       = 0;
            we.modified   = 0;

            if (is_file) {
                std::error_code sz_ec;
                auto sz = entry.file_size(sz_ec);
                if (!sz_ec) we.size = (int64_t) sz;
            }
            std::error_code tm_ec;
            auto mtime = entry.last_write_time(tm_ec);
            if (!tm_ec) we.modified = to_unix_seconds(mtime);

            out.push_back(std::move(we));

            if (is_dir && (int) frame.segs.size() + 1 < max_depth) {
                std::vector<std::string> child_segs = frame.segs;
                child_segs.push_back(name);
                stack.push_back({entry.path(), std::move(child_segs), hidden});
            }
        }
    }
}

// Single-entry cache for the last walk; queries arrive in bursts (one per
// keystroke) against a tree that rarely changes mid-burst, so a short TTL
// turns repeat searches into match+sort only. The mutex also serializes
// walks so concurrent requests cannot each trigger a full filesystem scan.
struct walk_cache {
    std::mutex mtx;
    std::string root;
    int max_depth = -1;
    std::chrono::steady_clock::time_point time;
    std::vector<walk_entry> entries;
};

walk_cache g_walk_cache;

constexpr int WALK_CACHE_TTL_MS = 3000;

} // namespace

bool search(
        const std::string & root,
        const search_options & opts,
        std::vector<search_entry> & results,
        std::string & err) {
    results.clear();

    std::lock_guard<std::mutex> lock(g_walk_cache.mtx);

    std::error_code ec;
    if (!fs::is_directory(root, ec) || ec) {
        err = "not a directory: " + root;
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const bool cache_fresh =
        g_walk_cache.root == root &&
        g_walk_cache.max_depth == opts.max_depth &&
        now - g_walk_cache.time < std::chrono::milliseconds(WALK_CACHE_TTL_MS);
    if (!cache_fresh) {
        walk_root(root, opts.max_depth, g_walk_cache.entries);
        g_walk_cache.root = root;
        g_walk_cache.max_depth = opts.max_depth;
        g_walk_cache.time = now;
    }
    const std::vector<walk_entry> & entries = g_walk_cache.entries;

    std::vector<std::string> q_segs = split_query_on_slash(opts.query);

    // Absolute path-like query ("/Users/foo/proj"): strip the root prefix so
    // pasting a full path under the root matches like the relative form.
    if (!opts.query.empty() && (opts.query[0] == '/' || opts.query[0] == '\\')) {
        const std::vector<std::string> root_segs = split_query_on_slash(root);
        const bool under_root = q_segs.size() >= root_segs.size() &&
            std::equal(root_segs.begin(), root_segs.end(), q_segs.begin(),
                [](const std::string & a, const std::string & b) {
                    return a.size() == b.size() && starts_with_ci(a, b);
                });
        if (under_root) {
            q_segs.erase(q_segs.begin(), q_segs.begin() + (ptrdiff_t) root_segs.size());
        }
    }

    struct scored {
        int tier;
        size_t idx;
    };
    std::vector<scored> matches;

    for (size_t i = 0; i < entries.size(); ++i) {
        const walk_entry & we = entries[i];

        if (opts.type == entry_type_filter::directory && !we.is_dir) continue;
        if (opts.type == entry_type_filter::file && we.is_dir) continue;
        if (!opts.show_hidden && we.hidden) continue;

        int tier = 0;
        if (!q_segs.empty()) {
            if (q_segs.size() >= 2) {
                tier = classify_pathlike(we.segs, q_segs);
            } else {
                tier = classify_basename(we.name, q_segs[0]);
            }
            const int max_tier = opts.match == match_mode::prefix ? 1 : 2;
            if (tier > max_tier) continue;
        }

        matches.push_back({tier, i});
    }

    // rank: tier (exact < prefix < substring), then non-hidden, shallower
    // first, most recently modified, alphabetical (case-insensitive)
    const auto cmp = [](const scored & a, const scored & b) {
        const walk_entry & x = entries[a.idx];
        const walk_entry & y = entries[b.idx];
        if (a.tier != b.tier) return a.tier < b.tier;
        if (x.hidden != y.hidden) return !x.hidden;
        if (x.depth != y.depth) return x.depth < y.depth;
        if (x.modified != y.modified) return x.modified > y.modified;
        return x.name_lower < y.name_lower;
    };
    if ((int) matches.size() > opts.limit) {
        std::partial_sort(matches.begin(), matches.begin() + opts.limit, matches.end(), cmp);
        matches.resize(opts.limit);
    } else {
        std::sort(matches.begin(), matches.end(), cmp);
    }

    results.reserve(matches.size());
    for (const auto & m : matches) {
        const walk_entry & we = entries[m.idx];
        search_entry e;
        e.name     = we.name;
        e.path     = we.path;
        e.parent   = we.parent;
        e.type     = we.is_dir ? "directory" : "file";
        e.size     = we.size;
        e.modified = we.modified;
        results.push_back(std::move(e));
    }

    return true;
}

// trim trailing whitespace so `.git/HEAD` lines from Windows checkouts don't
// leak '\r' into the branch name
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

    // `.git` typically sits a few levels above the cwd; the bounded walk
    // avoids probing every ancestor up to the filesystem root
    constexpr int MAX_DEPTH = 8;
    for (int depth = 0; depth <= MAX_DEPTH; ++depth) {
        const std::string cur_str = cur.string();

        // bail out as soon as the walk crosses outside the browse scope
        bool inside = false;
        for (const auto & root : allowed_roots) {
            if (is_child_of(cur_str, root)) {
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

        // standard layout: `.git/` is a directory containing HEAD, refs/, etc.
        if (fs::is_directory(git_dir, is_ec)) {
            std::ifstream head(git_dir / "HEAD");
            if (head) {
                std::string line;
                if (std::getline(head, line)) {
                    rstrip(line);
                    const std::string ref_prefix = "ref: refs/heads/";
                    info.is_repo = true;
                    info.root = cur_str;
                    if (line.rfind(ref_prefix, 0) == 0) {
                        info.branch = line.substr(ref_prefix.size());
                        if (info.branch.empty()) info.branch = "detached";
                    } else {
                        // detached HEAD (bare SHA), packed refs, partial clone, ...
                        info.branch = "detached";
                    }
                    return true;
                }
            }
            // `.git` exists but HEAD is missing/unreadable
            info.is_repo = true;
            info.root = cur_str;
            info.branch = "detached";
            return true;
        }

        // gitfile layout (submodules, worktrees): `.git` is a regular file
        // whose body is "gitdir: <path>"; the link is not chased since it can
        // reach outside the browse scope
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
