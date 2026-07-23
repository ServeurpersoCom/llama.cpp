#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Server-side filesystem search, used by the /filesystem/search endpoint.
// The endpoint is intended to back both the cwd picker in the chat form
// and inline file/folder mentions typed into the chat textarea. The UI
// sends raw text and the server returns ranked file/folder matches.

namespace server_fs {

enum class match_mode {
    substring,
    prefix,
};

enum class entry_type_filter {
    any,
    file,
    directory,
};

struct search_options {
    std::string query;
    // optional context root - when set, only entries inside this directory
    // (within an allowed root) are returned. used when the UI drills into a
    // previously-chosen folder (e.g. user typed "my-project/" and pressed a key).
    std::string context_path;
    match_mode match = match_mode::substring;
    entry_type_filter type = entry_type_filter::any;
    int limit = 50;
    int max_depth = 8;
    // hide entries whose path traverses (or whose basename is) a "."-prefixed
    // directory. UI exposes a checkbox so the user can opt in to seeing them.
    bool show_hidden = false;
};

struct search_entry {
    std::string name;            // basename, what the UI displays
    std::string path;            // absolute server-side path
    std::string parent;          // absolute path to the parent dir
    std::string type;            // "file" or "directory"
    int64_t size = 0;            // bytes, files only
    int64_t modified = 0;        // unix seconds since epoch
    int64_t added = 0;           // creation/birth time; falls back to modified
};

// Result of probing a path for git-repository metadata.
// `is_repo` is true once a `.git/` directory (or `.git` file) is found on
// the way up from the input path. `branch` is empty for the not-a-repo
// case; set to "detached" when `.git/HEAD` is a raw SHA rather than a
// refs/heads/* pointer. `sha` captures the HEAD hash when available.
struct git_info {
    bool is_repo = false;
    std::string root;            // absolute path to the directory holding `.git`
    std::string branch;          // branch name, "detached" for SHA-only HEAD, "" if not a repo
    std::string sha;             // current HEAD sha, empty when unreadable
};

// Compute the effective list of browse roots from the configured list.
// If `configured` is empty, defaults to a single root: $HOME (resolved
// against the process environment). Returns an empty vector and populates
// `err` if no valid root can be determined.
std::vector<std::string> effective_roots(
        const std::vector<std::string> & configured,
        std::string & err);

// Resolve `path` to a canonical absolute path inside one of `allowed_roots`.
// - empty path: first allowed root
// - relative path: resolved against the first allowed root
// - absolute path: canonicalized
// Returns the canonical absolute path on success, empty string on failure.
// The resolved path must be contained by some allowed root (with a separator
// boundary) - this is the path-traversal guard. Backend errors populate `err`.
std::string resolve_path(
        const std::string & path,
        const std::vector<std::string> & allowed_roots,
        std::string & err);

// Walk `root` and populate `results` with entries matching `opts`.
// `root` must be the canonical absolute path of an existing directory.
// Returns false on error.
bool search(
        const std::string & root,
        const search_options & opts,
        std::vector<search_entry> & results,
        std::string & err);

// Walk up from `path` looking for `.git/` (directory form) or `.git`
// (gitfile form, treated as a submodule marker). On every candidate
// parent, also confirm the directory lies inside one of `allowed_roots` -
// the walk is capped before leaving the configured browse scope so a
// malicious path input cannot probe the host filesystem outside the
// sandbox. When a match is found, populate `info` with the repo metadata
// and return true. Returns false with `info` left default when no
// repository can be found within the walk depth limit.
bool git_status(
        const std::string & path,
        const std::vector<std::string> & allowed_roots,
        git_info & info,
        std::string & err);

} // namespace server_fs
