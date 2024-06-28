#!/usr/bin/env bash

# Define a list of valid strings
submodules=("")

# Iterate over the contents of the directory
for FILE in src/trtutils/*;
do
    # Extract the filename from the full path
    filename=$(basename "$FILE")
    
    # Check if the filename is in the list of valid strings
    if [[ " ${submodules[*]} " =~ " $filename " ]]; then
        echo "Running $filename..."
        python3 -m mypy --follow-imports=silent $FILE
    else
        echo "Skipping $filename..."
    fi
done
