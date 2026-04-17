#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Aura Source Export Script
# Creates text-file parts (3.9M chars each) + folder copy (top 1000 files)
###############################################################################

AURA_ROOT="$HOME/Desktop/aura"
OUT_DIR="$HOME/Downloads"
COPY_DIR="$OUT_DIR/aura_source_copy"
CHAR_LIMIT=3900000
HEADER_TEMPLATE="═══════════════════════════════════════════════════════════════"

# ── Step 0: Clean previous exports ──────────────────────────────────────────
echo "==> Cleaning previous exports..."
rm -f "$OUT_DIR"/aura_source_part_*.txt
rm -rf "$COPY_DIR"
mkdir -p "$COPY_DIR"

# ── Step 1: Find all architecture files ─────────────────────────────────────
# Extensions: .py .yaml .yml .json .toml .sh .cfg .ini .conf
# Plus: Dockerfile*, Makefile*, docker-compose*
# Exclude: __pycache__, .git, node_modules, venv, .venv, build, dist,
#          .egg, archive, .tox, models_gguf, models/, test_data, test_vdb,
#          test_brain, media files, model weights, data/, storage/,
#          memory_store/, scratch/

TMPFILE=$(mktemp)

find "$AURA_ROOT" -type f \( \
    -name "*.py" -o \
    -name "*.yaml" -o -name "*.yml" -o \
    -name "*.json" -o \
    -name "*.toml" -o \
    -name "*.sh" -o \
    -name "*.cfg" -o -name "*.ini" -o -name "*.conf" -o \
    -name "Dockerfile" -o -name "Dockerfile.*" -o \
    -name "Makefile" -o -name "Makefile.*" -o \
    -name "docker-compose*" -o \
    -name "*.rs" -o \
    -name "*.proto" \
    \) \
    ! -path "*/__pycache__/*" \
    ! -path "*/.git/*" \
    ! -path "*/node_modules/*" \
    ! -path "*/venv/*" \
    ! -path "*/.venv/*" \
    ! -path "*/build/*" \
    ! -path "*/dist/*" \
    ! -path "*/.egg*" \
    ! -path "*/archive/*" \
    ! -path "*/.tox/*" \
    ! -path "*/models_gguf/*" \
    ! -path "*/models/*" \
    ! -path "*/test_data/*" \
    ! -path "*/test_vdb/*" \
    ! -path "*/test_brain/*" \
    ! -path "*/data/*" \
    ! -path "*/storage/*" \
    ! -path "*/memory_store/*" \
    ! -path "*/scratch/*" \
    ! -path "*/assets/*" \
    ! -path "*/.aura/*" \
    ! -path "*/qa_reports/*" \
    ! -path "*/experiments/*" \
    ! -name "*.pyc" \
    ! -name "*.pyo" \
    ! -name "*.so" \
    ! -name "*.dylib" \
    ! -name "*.whl" \
    ! -name "*.zip" \
    ! -name "*.tar.*" \
    ! -name "*.gguf" \
    ! -name "*.bin" \
    ! -name "*.pkl" \
    ! -name "*.npy" \
    ! -name "*.npz" \
    ! -name "*.h5" \
    ! -name "*.pt" \
    ! -name "*.pth" \
    ! -name "*.onnx" \
    ! -name "export_source.sh" \
    | sort > "$TMPFILE"

TOTAL_FILES=$(wc -l < "$TMPFILE")
echo "==> Found $TOTAL_FILES architecture files"

# ── Step 2: Assign priority tiers ──────────────────────────────────────────
# Tier 1 (highest): core/ Python files
# Tier 2: aura/, aura_main/ - main entry points
# Tier 3: brain/, consciousness/, llm/ - intelligence subsystems
# Tier 4: interface/, senses/, skills/, tools/ - interaction layer
# Tier 5: autonomy_engine/, optimizer/, proof_kernel/ - advanced subsystems
# Tier 6: infrastructure/, cloud/, security/, executors/ - infra
# Tier 7: integration/, utils/, memory/ - support
# Tier 8: scripts/, systemd/, docker/, benchmarks/ - ops
# Tier 9: tests/, training/, research/, demos/ - secondary
# Tier 10: root-level files, configs
# Tier 11: remaining .json files (often data-like)

SORTED_FILE=$(mktemp)

assign_priority() {
    local f="$1"
    local rel="${f#$AURA_ROOT/}"

    # .py files get higher base priority than configs
    local is_py=0
    [[ "$rel" == *.py ]] && is_py=1

    case "$rel" in
        core/*.py|core/**/*.py)                    echo "01 $f" ;;
        core/*)                                     echo "06 $f" ;;
        aura_main.py|aura/*.py|aura_main/*.py)     echo "02 $f" ;;
        aura/*|aura_main/*)                         echo "07 $f" ;;
        llm/*.py|llm/**/*.py)                       echo "03 $f" ;;
        llm/*)                                      echo "08 $f" ;;
        interface/*.py|interface/**/*.py)            echo "04 $f" ;;
        interface/*)                                echo "08 $f" ;;
        senses/*.py|senses/**/*.py)                 echo "04 $f" ;;
        senses/*)                                   echo "08 $f" ;;
        skills/*.py|skills/**/*.py)                 echo "04 $f" ;;
        skills/*)                                   echo "08 $f" ;;
        tools/*.py|tools/**/*.py)                   echo "04 $f" ;;
        tools/*)                                    echo "08 $f" ;;
        autonomy_engine/*.py|autonomy_engine/**/*.py) echo "05 $f" ;;
        autonomy_engine/*)                          echo "09 $f" ;;
        optimizer/*.py|optimizer/**/*.py)            echo "05 $f" ;;
        optimizer/*)                                echo "09 $f" ;;
        proof_kernel/*.py|proof_kernel/**/*.py)     echo "05 $f" ;;
        proof_kernel/*.rs|proof_kernel/**/*.rs)     echo "05 $f" ;;
        proof_kernel/*)                             echo "09 $f" ;;
        infrastructure/*.py|infrastructure/**/*.py) echo "06 $f" ;;
        infrastructure/*)                           echo "09 $f" ;;
        cloud/*.py|cloud/**/*.py)                   echo "06 $f" ;;
        cloud/*)                                    echo "09 $f" ;;
        security/*.py|security/**/*.py)             echo "06 $f" ;;
        security/*)                                 echo "09 $f" ;;
        executors/*.py|executors/**/*.py)            echo "06 $f" ;;
        executors/*)                                echo "09 $f" ;;
        integration/*.py|integration/**/*.py)       echo "07 $f" ;;
        integration/*)                              echo "09 $f" ;;
        utils/*.py|utils/**/*.py)                   echo "07 $f" ;;
        utils/*)                                    echo "09 $f" ;;
        memory/*.py|memory/**/*.py)                 echo "07 $f" ;;
        memory/*)                                   echo "09 $f" ;;
        rust_extensions/*.rs|rust_extensions/**/*.rs) echo "07 $f" ;;
        rust_extensions/*)                          echo "09 $f" ;;
        scripts/*|systemd/*|docker/*|benchmarks/*)  echo "10 $f" ;;
        tests/*.py|tests/**/*.py)                   echo "11 $f" ;;
        tests/*)                                    echo "12 $f" ;;
        training/*.py|training/**/*.py)             echo "11 $f" ;;
        training/*)                                 echo "12 $f" ;;
        research/*.py|research/**/*.py)             echo "11 $f" ;;
        research/*)                                 echo "12 $f" ;;
        demos/*.py|demos/**/*.py)                   echo "11 $f" ;;
        demos/*)                                    echo "12 $f" ;;
        audit/*.py|audit/**/*.py)                   echo "11 $f" ;;
        audit/*)                                    echo "12 $f" ;;
        # Root-level .py files
        *.py)                                       echo "08 $f" ;;
        # Root-level configs
        *.yaml|*.yml|*.toml|*.cfg|*.ini|*.conf)    echo "09 $f" ;;
        # Dockerfiles, Makefiles at root
        Dockerfile*|Makefile*|docker-compose*)      echo "09 $f" ;;
        # Root-level .sh
        *.sh)                                       echo "10 $f" ;;
        # Root-level .json (often package locks, etc.)
        *.json)                                     echo "13 $f" ;;
        # Everything else
        *)                                          echo "14 $f" ;;
    esac
}

echo "==> Sorting files by architectural importance..."
while IFS= read -r filepath; do
    assign_priority "$filepath"
done < "$TMPFILE" | sort -t' ' -k1,1n -k2 | cut -d' ' -f2- > "$SORTED_FILE"

# ── Step 3: Generate text parts with 3.9M char limit ───────────────────────
echo "==> Generating text export parts..."

PART_NUM=1
CURRENT_CHARS=0
CURRENT_FILE="$OUT_DIR/aura_source_part_${PART_NUM}.txt"
FILE_COUNT=0
TOTAL_CHARS=0

# Start first file
> "$CURRENT_FILE"

while IFS= read -r filepath; do
    # Get relative path
    rel_path="${filepath#$AURA_ROOT/}"

    # Build the header + content block
    header="${HEADER_TEMPLATE}
FILE: ${rel_path}
${HEADER_TEMPLATE}
"
    # Read file content
    content=$(cat "$filepath" 2>/dev/null || echo "[BINARY OR UNREADABLE]")
    block="${header}${content}

"
    block_len=${#block}

    # Check if we need a new part
    if (( CURRENT_CHARS + block_len > CHAR_LIMIT && CURRENT_CHARS > 0 )); then
        echo "    Part $PART_NUM: $CURRENT_CHARS chars"
        PART_NUM=$((PART_NUM + 1))
        CURRENT_FILE="$OUT_DIR/aura_source_part_${PART_NUM}.txt"
        > "$CURRENT_FILE"
        CURRENT_CHARS=0
    fi

    # Append to current part
    printf '%s' "$block" >> "$CURRENT_FILE"
    CURRENT_CHARS=$((CURRENT_CHARS + block_len))
    TOTAL_CHARS=$((TOTAL_CHARS + block_len))
    FILE_COUNT=$((FILE_COUNT + 1))

done < "$SORTED_FILE"

echo "    Part $PART_NUM: $CURRENT_CHARS chars"
echo ""
echo "==> Text export complete:"
echo "    Files exported: $FILE_COUNT"
echo "    Total characters: $TOTAL_CHARS"
echo "    Parts created: $PART_NUM"

# Show part sizes
echo ""
echo "==> Part sizes:"
for p in "$OUT_DIR"/aura_source_part_*.txt; do
    sz=$(wc -c < "$p")
    name=$(basename "$p")
    echo "    $name: $sz bytes"
done

# ── Step 4: Create folder copy with top 1000 files ─────────────────────────
echo ""
echo "==> Creating folder copy (top 1000 files by importance)..."

COPY_COUNT=0
while IFS= read -r filepath; do
    if (( COPY_COUNT >= 1000 )); then
        break
    fi
    rel_path="${filepath#$AURA_ROOT/}"
    dest_dir="$COPY_DIR/$(dirname "$rel_path")"
    mkdir -p "$dest_dir"
    cp "$filepath" "$dest_dir/"
    COPY_COUNT=$((COPY_COUNT + 1))
done < "$SORTED_FILE"

echo "    Files copied: $COPY_COUNT"
COPY_SIZE=$(du -sh "$COPY_DIR" | cut -f1)
echo "    Total size: $COPY_SIZE"

# ── Cleanup ─────────────────────────────────────────────────────────────────
rm -f "$TMPFILE" "$SORTED_FILE"

echo ""
echo "==> Export complete!"
echo "    Text parts: $OUT_DIR/aura_source_part_*.txt"
echo "    Folder copy: $COPY_DIR/"
