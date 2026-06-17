#!/usr/bin/env bash
# Deploy lucid-mt3 on the Central Command host. Idempotent + self-verifying.
#
# Run this ON the CC host (failsafe@10.205.10.16), from the lucid-mt3 directory
# inside the lucid-central-command checkout:
#
#   cd ~/lucid-central-command/lucid-mt3 && ./deploy-cc.sh
#
# Prereqs on the host: docker + nvidia runtime, git, and the CC stack already up
# (EMQX + auth running) so we can provision the mt3 MQTT identity.
set -euo pipefail

ONCE_REPO="${ONCE_REPO:-https://github.com/IERoboticsAILab/once-project.git}"
ONCE_REF="${ONCE_REF:-gpu-accelerated}"
ONCE_DIR="${ONCE_DIR:-$HOME/once-project}"
LEARN_DIR="$ONCE_DIR/1000_tasks/learning_thousand_tasks"
DEMOS_DST="/var/lib/lucid-mt3/demonstrations"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UMBRELLA="$(cd "$HERE/.." && pwd)"

log() { printf '\n=== %s ===\n' "$*"; }

# 1. Get / refresh the once-project checkout (source for the base image + demos).
log "1/6 once-project checkout ($ONCE_REF)"
if [ -d "$ONCE_DIR/.git" ]; then
  git -C "$ONCE_DIR" fetch --depth 1 origin "$ONCE_REF"
  git -C "$ONCE_DIR" checkout -q "$ONCE_REF"
  git -C "$ONCE_DIR" reset --hard -q "origin/$ONCE_REF"
else
  git clone --depth 1 --branch "$ONCE_REF" "$ONCE_REPO" "$ONCE_DIR"
fi

# 2. Build the thousand-tasks base image (heavy; cached after first run).
log "2/6 build thousand-tasks base image"
docker build -t thousand-tasks "$LEARN_DIR"

# 3. Pre-seed demonstrations into the persistent host store (once).
log "3/6 pre-seed demonstrations -> $DEMOS_DST"
sudo mkdir -p "$DEMOS_DST"
if [ -z "$(ls -A "$DEMOS_DST" 2>/dev/null)" ]; then
  sudo cp -a "$LEARN_DIR/assets/demonstrations/." "$DEMOS_DST/"
  echo "seeded $(ls "$DEMOS_DST" | wc -l) demos"
else
  echo "store already populated ($(ls "$DEMOS_DST" | wc -l) entries) - leaving as-is"
fi

# 4. Provision the mt3 MQTT identity and write .env (only if not already set).
log "4/6 MQTT credentials (.env)"
if [ -f "$HERE/.env" ] && grep -q '^MT3_MQTT_PASSWORD=.\+' "$HERE/.env" \
   && ! grep -q '^MT3_MQTT_PASSWORD=change-me' "$HERE/.env"; then
  echo ".env already has MT3_MQTT_PASSWORD - skipping provisioning"
else
  echo "Provisioning agent 'mt3' on the auth service..."
  echo ">>> Run this, then paste the printed password into $HERE/.env (MT3_MQTT_PASSWORD):"
  echo "      (cd $UMBRELLA && docker compose exec auth python manage.py add-agent mt3)"
  echo ">>> Then re-run this script. Aborting so you can set the password."
  cp -n "$HERE/.env.example" "$HERE/.env" 2>/dev/null || true
  exit 2
fi

# 5. Build + start the service.
log "5/6 build + start lucid-mt3"
( cd "$HERE" && docker compose up -d --build )

# 6. Verify.
log "6/6 verify"
sleep 4
docker ps --filter name=lucid-mt3 --format 'table {{.Names}}\t{{.Status}}'
echo "--- recent logs ---"
docker logs --tail 20 lucid-mt3 2>&1 || true
if docker logs lucid-mt3 2>&1 | grep -q "Subscribing to run + upload_demo"; then
  echo
  echo "OK: lucid-mt3 connected to MQTT and subscribed."
else
  echo
  echo "WARN: did not see the subscribe line yet. Check 'docker logs -f lucid-mt3'."
  echo "      (common causes: bad MQTT_PASSWORD, broker not reachable on localhost:1883)"
  exit 1
fi
