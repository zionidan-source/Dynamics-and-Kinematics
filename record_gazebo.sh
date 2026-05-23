#!/usr/bin/env bash
# record_gazebo.sh — Wayland-safe wrapper around GNOME's built-in screencast.
set -e

OUTPUT="${1:-gazebo_motion.mp4}"
SCREENCAST_DIR="$HOME/Videos/Screencasts"

if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg is required. Install with: sudo apt install ffmpeg"
    exit 1
fi

# GNOME 46+ removed the screencast length cap, so no gsettings tweak needed.

mkdir -p "$SCREENCAST_DIR"
BEFORE_FILES=$(mktemp)
ls -1 "$SCREENCAST_DIR" 2>/dev/null | sort > "$BEFORE_FILES"

echo "================================================================"
echo " UR5 Gazebo recording (Wayland-safe via GNOME built-in)"
echo "================================================================"
echo ""
echo " 1. Make sure Gazebo is already running and the UR5 is visible."
echo " 2. Press Ctrl+Alt+Shift+R to START recording (red dot top-right)."
echo " 3. In another terminal: python3 simulation_gazebo.py"
echo " 4. When the motion ends, press Ctrl+Alt+Shift+R again to STOP."
echo " 5. Come back to THIS terminal and press Enter."
echo ""
read -r -p "Press Enter once you have stopped the recording... " _

AFTER_FILES=$(mktemp)
ls -1 "$SCREENCAST_DIR" 2>/dev/null | sort > "$AFTER_FILES"
NEW=$(comm -13 "$BEFORE_FILES" "$AFTER_FILES" | head -1)
rm -f "$BEFORE_FILES" "$AFTER_FILES"

if [[ -z "$NEW" ]]; then
    echo "ERROR: no new screencast found in $SCREENCAST_DIR."
    echo "       Try: find ~ -name '*.webm' -mmin -10"
    exit 1
fi

SRC="$SCREENCAST_DIR/$NEW"
echo "Found: $NEW ($(du -h "$SRC" | cut -f1))"
echo "Converting to MP4..."
ffmpeg -hide_banner -loglevel error -y \
       -i "$SRC" \
       -c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p \
       "$OUTPUT"

echo ""
echo "Done. Output: $(pwd)/$OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
echo "Original .webm kept at: $SRC"
