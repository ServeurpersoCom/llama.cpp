#!/bin/bash
# Build wrapper for llama.cpp webui (filters KaTeX Sass deprecation warnings)
#
# KaTeX hasn't migrated to Dart Sass 3.0 syntax yet (still uses @import and global functions).
# This script filters out KaTeX-specific deprecation warnings during build to keep output clean.
# When KaTeX releases a version compatible with Dart Sass 3.0, this filtering can be removed.
# Latest KaTeX version checked: 0.16.25 (October 2025) - still not migrated.

# KaTeX Sass deprecation warnings to filter out
KATEX_IMPORT_WARNING="DEPRECATION WARNING \[import\]: Sass @import rules are deprecated and will be removed in Dart Sass 3.0.0."
KATEX_BUILTIN_WARNING="DEPRECATION WARNING \[global-builtin\]: Global built-in functions are deprecated and will be removed in Dart Sass 3.0.0."

# Force color output even when piping
export FORCE_COLOR=1

vite build 2>&1 | awk -v import_warn="$KATEX_IMPORT_WARNING" -v builtin_warn="$KATEX_BUILTIN_WARNING" '
    $0 ~ import_warn || $0 ~ builtin_warn { skip=1; next }
    skip && /root stylesheet/ {
        skip=0
        getline
        if (NF > 0) print
        next
    }
    skip { next }
    !skip { print }
' && ./scripts/post-build.sh
