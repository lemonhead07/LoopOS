#!/bin/bash
# Concatenate Wikipedia shard files into a single corpus that the streaming loader can consume.
# Usage: ./scripts/make_wiki_corpus.sh [source_dir] [output_file] [max_files]
#   source_dir  - Directory containing shard files (default: data/pretraining/wiki/fullEnglish)
#   output_file - Destination flattened corpus (default: data/pretraining/wiki/wiki_corpus.txt)
#   max_files   - Optional limit on number of files to concatenate (0 = all)

set -euo pipefail

SOURCE_DIR=${1:-"data/pretraining/wiki/fullEnglish"}
OUTPUT_FILE=${2:-"data/pretraining/wiki/wiki_corpus.txt"}
MAX_FILES=${3:-0}

if [ ! -d "$SOURCE_DIR" ]; then
    echo "[make_wiki_corpus] Source directory not found: $SOURCE_DIR" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

TMP_FILE="${OUTPUT_FILE}.tmp"
rm -f "$TMP_FILE"
touch "$TMP_FILE"

COUNTER=0
TOTAL_FILES=0
mapfile -t FILES < <(find "$SOURCE_DIR" -type f -name 'wiki_*' -print | LC_ALL=C sort)
TOTAL_FILES=${#FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "[make_wiki_corpus] No wiki_* shards found under $SOURCE_DIR" >&2
    rm -f "$TMP_FILE"
    exit 1
fi

echo "[make_wiki_corpus] Flattening $TOTAL_FILES files from $SOURCE_DIR"

for FILE in "${FILES[@]}"; do
    if [ "$MAX_FILES" -gt 0 ] && [ "$COUNTER" -ge "$MAX_FILES" ]; then
        break
    fi

    echo "[make_wiki_corpus] Appending $(basename "$FILE")" >&2
    cat "$FILE" >> "$TMP_FILE"

    # Ensure shard boundaries become paragraph breaks
    printf '\n' >> "$TMP_FILE"

    COUNTER=$((COUNTER + 1))

done

mv "$TMP_FILE" "$OUTPUT_FILE"

echo "[make_wiki_corpus] Wrote $(du -h "$OUTPUT_FILE" | awk '{print $1}') to $OUTPUT_FILE" >&2
if [ "$MAX_FILES" -gt 0 ]; then
    echo "[make_wiki_corpus] Included $COUNTER shard(s)" >&2
else
    echo "[make_wiki_corpus] Included all $COUNTER shard(s)" >&2
fi
