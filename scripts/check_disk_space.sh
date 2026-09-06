#!/bin/bash
# Check free space in the ext4 WSL filesystem before downloading models

REPO_DIR="/mnt/c/Users/brian/Desktop/projects/niel_landa"
MIN_SPACE_GB=20

# Get free space in GB
FREE_SPACE=$(df "$REPO_DIR" | awk 'NR==2 {print int($4/1024/1024)}')

echo "Disk space check for $REPO_DIR"
echo "Free space: ${FREE_SPACE} GB"
echo "Required: ${MIN_SPACE_GB} GB"

if [ "$FREE_SPACE" -lt "$MIN_SPACE_GB" ]; then
    echo "ERROR: Insufficient free space. Need at least ${MIN_SPACE_GB} GB, have ${FREE_SPACE} GB."
    exit 1
else
    echo "OK: Sufficient space available."
    exit 0
fi
