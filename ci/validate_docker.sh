#!/usr/bin/env bash
set -euo pipefail

# Validate Docker test containers by building them and verifying
# that all expected packages are importable.
#
# Usage:
#   ./ci/validate_docker.sh          # validate all CUDA versions
#   ./ci/validate_docker.sh 12       # validate only cu12
#   ./ci/validate_docker.sh 11 13    # validate cu11 and cu13

COMPOSE="docker compose -f docker/docker-compose.test.yml"
ALL_VERSIONS=("11" "12" "13")

if [[ $# -gt 0 ]]; then
    VERSIONS=("$@")
else
    VERSIONS=("${ALL_VERSIONS[@]}")
fi

# Validate version arguments
for VER in "${VERSIONS[@]}"; do
    case "$VER" in
        11|12|13) ;;
        *)
            echo "Error: invalid CUDA version '${VER}'. Must be one of: 11, 12, 13"
            exit 1
            ;;
    esac
done

EXIT_CODE=0

for VER in "${VERSIONS[@]}"; do
    SERVICE="test-cu${VER}"
    echo "========================================="
    echo "  Validating ${SERVICE}"
    echo "========================================="

    # Build the image
    echo "--- Building image ---"
    if ! $COMPOSE build "$SERVICE"; then
        echo "FAIL: ${SERVICE} image failed to build"
        EXIT_CODE=1
        continue
    fi

    # Verify Python packages are importable
    echo "--- Checking Python packages ---"
    VALIDATION_SCRIPT=$(cat <<'PYEOF'
import sys

errors = []
packages = [
    "trtutils",
    "tensorrt",
    "cuda",
    "pytest",
    "cv2",
    "numpy",
    "cv2ext",
    "tqdm",
    "typing_extensions",
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"  OK: {pkg}")
    except ImportError as e:
        print(f"  FAIL: {pkg} ({e})")
        errors.append(pkg)

if errors:
    print(f"\nFailed to import: {', '.join(errors)}")
    sys.exit(1)

print("\nAll Python packages imported successfully.")
PYEOF
    )

    if ! $COMPOSE run --rm "$SERVICE" python3 -c "$VALIDATION_SCRIPT"; then
        echo "FAIL: ${SERVICE} Python package validation failed"
        EXIT_CODE=1
    fi

    # Verify CLI tools
    echo "--- Checking CLI tools ---"
    CLI_CHECK=$(cat <<'SHEOF'
exit_code=0
for tool in ruff ty; do
    if command -v "$tool" > /dev/null 2>&1; then
        version=$("$tool" --version 2>&1 || true)
        echo "  OK: $tool ($version)"
    else
        echo "  FAIL: $tool not found"
        exit_code=1
    fi
done
exit $exit_code
SHEOF
    )

    if ! $COMPOSE run --rm "$SERVICE" bash -c "$CLI_CHECK"; then
        echo "FAIL: ${SERVICE} CLI tool validation failed"
        EXIT_CODE=1
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo "=== ${SERVICE}: PASSED ==="
    else
        echo "=== ${SERVICE}: FAILED ==="
    fi
    echo
done

if [ $EXIT_CODE -eq 0 ]; then
    echo "All containers validated successfully."
else
    echo "Some containers failed validation."
fi

exit $EXIT_CODE
