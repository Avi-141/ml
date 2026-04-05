#!/bin/zsh

set -euo pipefail

SOURCE_DIR="<>"
TARGET_DIR="<>"

show_help() {
  cat <<'EOF'
Usage: ./sync_notes_to_obsidian.sh [options]

Sync this repo's notes folder into the Obsidian vault folder:

Options:
  --dry-run   Show what would change without modifying files
  --delete    Mirror deletions from repo notes into the vault target
  --help      Show this help message

Examples:
  ./sync_notes_to_obsidian.sh --dry-run
  ./sync_notes_to_obsidian.sh
  ./sync_notes_to_obsidian.sh --delete
EOF
}

dry_run=false
delete_mode=false

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      dry_run=true
      ;;
    --delete)
      delete_mode=true
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Run with --help to see valid options." >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source notes folder not found: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

cmd=(
  rsync
  -av
  --exclude
  ".DS_Store"
)

if [[ "$dry_run" == true ]]; then
  cmd+=(--dry-run)
fi

if [[ "$delete_mode" == true ]]; then
  cmd+=(--delete)
fi

cmd+=("$SOURCE_DIR" "$TARGET_DIR")

echo "Syncing notes:"
echo "  source: $SOURCE_DIR"
echo "  target: $TARGET_DIR"

if [[ "$dry_run" == true ]]; then
  echo "  mode: dry run"
elif [[ "$delete_mode" == true ]]; then
  echo "  mode: mirror with delete"
else
  echo "  mode: copy/update only"
fi

"${cmd[@]}"

echo "Done."
