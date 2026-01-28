#!/usr/bin/env bash
set -euo pipefail

# Wrapper for running GitHub Actions locally that supports:
# - nektos/act binary: `act ...`
# - GitHub CLI integration: `gh act ...`
#
# Optional override:
# - ACT_BIN: force a specific `act` binary (e.g. /usr/local/bin/act)

if [[ -n "${ACT_BIN:-}" ]]; then
  if ! command -v "${ACT_BIN}" >/dev/null 2>&1; then
    echo "error: ACT_BIN is set but not found on PATH: ${ACT_BIN}" >&2
    exit 127
  fi
  exec "${ACT_BIN}" "$@"
fi

if command -v act >/dev/null 2>&1; then
  exec act "$@"
fi

if command -v gh >/dev/null 2>&1; then
  # `gh act` works when the act extension/plugin is available in gh.
  if gh act --version >/dev/null 2>&1; then
    exec gh act "$@"
  fi
fi

cat >&2 <<'EOF'
error: could not find a local runner for GitHub Actions.

Tried, in order:
  - `act` (nektos/act)
  - `gh act` (GitHub CLI integration/extension)

Install options:
  - Install nektos/act so `act` is on PATH
  - Or install GitHub CLI and ensure `gh act` works on this machine

You can also set ACT_BIN to point to a specific `act` executable.
EOF
exit 127
