#!/usr/bin/env bash
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
#
# Check and auto-update copyright headers in Python files.
#
# Usage:
#   ci/check_copyright.sh          # Pre-commit mode: check staged .py files
#   ci/check_copyright.sh --check  # Check-only mode: scan src/, tests/, ci/
set -euo pipefail

CURRENT_YEAR=$(date +%Y)
FAIL=0

# ---------- helpers ----------------------------------------------------------

# Get the line number where the copyright should appear (1 normally, 2 after shebang)
copyright_line() {
    local file="$1"
    local first_line
    first_line=$(head -n1 "$file")
    if [[ "$first_line" == "#!"* ]]; then
        echo 2
    else
        echo 1
    fi
}

# Check if a file has a copyright header at the expected line
has_copyright() {
    local file="$1"
    local lineno
    lineno=$(copyright_line "$file")
    local line
    line=$(sed -n "${lineno}p" "$file")
    [[ "$line" == "# Copyright (c)"* ]]
}

# Check if the copyright line already contains the current year
has_current_year() {
    local file="$1"
    local lineno
    lineno=$(copyright_line "$file")
    local line
    line=$(sed -n "${lineno}p" "$file")
    [[ "$line" == *"$CURRENT_YEAR"* ]]
}

# Update the copyright year in-place. Handles:
#   2024       -> 2024-2026
#   2024-2025  -> 2024-2026
update_year() {
    local file="$1"
    local lineno
    lineno=$(copyright_line "$file")
    # Replace a single year or year-range with year-CURRENT_YEAR
    # Single year: "Copyright (c) 2024 " -> "Copyright (c) 2024-2026 "
    # Range:       "Copyright (c) 2024-2025 " -> "Copyright (c) 2024-2026 "
    sed -i "${lineno}s/Copyright (c) \([0-9]\{4\}\)\(-[0-9]\{4\}\)\{0,1\} /Copyright (c) \1-${CURRENT_YEAR} /" "$file"
}

# ---------- modes ------------------------------------------------------------

run_precommit() {
    local files
    files=$(git diff --cached --name-only --diff-filter=ACM -- '*.py') || true
    if [[ -z "$files" ]]; then
        return 0
    fi

    while IFS= read -r file; do
        [[ -f "$file" ]] || continue

        if ! has_copyright "$file"; then
            echo "ERROR: missing copyright header: $file"
            FAIL=1
            continue
        fi

        if ! has_current_year "$file"; then
            update_year "$file"
            if has_current_year "$file"; then
                git add "$file"
                echo "UPDATED: copyright year in $file"
            else
                echo "ERROR: failed to update copyright year: $file"
                FAIL=1
            fi
        fi
    done <<< "$files"
}

run_check() {
    local files
    files=$(find src/ tests/ ci/ -name '*.py' -type f 2>/dev/null | sort) || true
    if [[ -z "$files" ]]; then
        return 0
    fi

    while IFS= read -r file; do
        [[ -f "$file" ]] || continue

        if ! has_copyright "$file"; then
            echo "ERROR: missing copyright header: $file"
            FAIL=1
            continue
        fi

        if ! has_current_year "$file"; then
            echo "STALE: copyright year needs update: $file"
            FAIL=1
        fi
    done <<< "$files"
}

# ---------- main -------------------------------------------------------------

case "${1:-}" in
    --check)
        run_check
        ;;
    *)
        run_precommit
        ;;
esac

if [[ "$FAIL" -ne 0 ]]; then
    echo ""
    echo "Copyright header check failed. Add missing headers in this format:"
    echo "  # Copyright (c) $CURRENT_YEAR \$AUTHOR_NAME (\$AUTHOR_EMAIL)"
    echo "  #"
    echo "  # MIT License"
    exit 1
fi

exit 0
