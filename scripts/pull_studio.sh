#!/bin/bash
# Pull Studio inference artifacts into work/ on the mini.
set -euo pipefail
cd "$(dirname "$0")/.."
KEY=~/.ssh/id_ed25519_mac_studio_simulations
R="luis@100.74.208.47"
SSH="ssh -i $KEY -o BatchMode=yes"
mkdir -p work/track work/masks work/pose3d
rsync -a -e "$SSH" "$R:~/fightlab/work/track/fighters.json" work/track/
rsync -a -e "$SSH" "$R:~/fightlab/work/masks/" work/masks/
rsync -a -e "$SSH" "$R:~/fightlab/work/pose3d/" work/pose3d/ 2>/dev/null || echo "(no pose3d yet)"
rsync -a -e "$SSH" "$R:~/fightlab/work/0*.log" work/ 2>/dev/null || true
echo pulled
