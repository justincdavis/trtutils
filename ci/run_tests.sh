#!/usr/bin/env bash

# Set the path to the tests directory
test_directory="./tests"

# Loop through all Python files in the tests directory
for file in "$test_directory"/*.py; do
    echo "Running pytest on $file"
    python3 -m pytest "$file"
done
