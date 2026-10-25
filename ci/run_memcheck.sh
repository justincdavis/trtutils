#!/usr/bin/env bash
# Copyright (c) 2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License

# Run pytest under compute-sanitizer memcheck and summarize the errors by kind.
# Exits 1 when memcheck reports any error.
set -euo pipefail

usage() {
    echo "Usage: $0 [pytest args...]"
    echo "  Default pytest args: tests/core"
    echo "  Env: PYTHON (default python3), MEMCHECK_LOG (default memcheck.log)"
}

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

SANITIZER="$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer)"
if [ ! -x "$SANITIZER" ]; then
    echo "compute-sanitizer not found on PATH or in /usr/local/cuda/bin" >&2
    exit 2
fi

LOG="${MEMCHECK_LOG:-memcheck.log}"
if [ "$#" -eq 0 ]; then
    set -- tests/core
fi

# pytest's own exit code is ignored here, failures show in its output
"$SANITIZER" --tool memcheck --print-limit 0 --log-file "$LOG" \
    "${PYTHON:-python3}" -m pytest -q -p no:cacheprovider "$@" || true

ERRORS="$(sed -n 's/^========= ERROR SUMMARY: \([0-9]*\) error.*/\1/p' "$LOG" | tail -n 1)"
echo "memcheck: ${ERRORS:-?} errors (log: $LOG)"
grep '^========= [A-Z]' "$LOG" | grep -v -e 'ERROR SUMMARY' -e 'COMPUTE-SANITIZER' \
    | sed 's/ at .*//' | sort | uniq -c | sort -rn || true
[ "${ERRORS:-1}" -eq 0 ]
