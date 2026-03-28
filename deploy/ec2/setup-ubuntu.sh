#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-/opt/aura}"
REPO_URL="${2:-}"
BRANCH="${3:-main}"

if [ -z "$REPO_URL" ]; then
  echo "Usage: ./deploy/ec2/setup-ubuntu.sh /opt/aura <repo-url> [branch]"
  exit 1
fi

sudo apt-get update
sudo apt-get install -y ca-certificates curl git

if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi

sudo usermod -aG docker "$USER"

sudo mkdir -p "$APP_DIR"
sudo chown -R "$USER":"$USER" "$APP_DIR"

if [ ! -d "$APP_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"

if [ ! -f .env.production ]; then
  cp .env.example .env.production
  echo "Created .env.production from .env.example. Update it before deploy."
fi

echo "Setup finished. Re-login to apply docker group, then run:"
echo "bash deploy/ec2/deploy-ec2.sh $APP_DIR $BRANCH"
