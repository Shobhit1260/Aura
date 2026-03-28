#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-/opt/aura}"
BRANCH="${2:-main}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed. Install Docker before running deploy."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose plugin is not installed."
  exit 1
fi

if [ ! -d "$APP_DIR/.git" ]; then
  echo "Git repository not found at $APP_DIR"
  echo "Clone the repository first, then re-run deploy."
  exit 1
fi

cd "$APP_DIR"

echo "Updating source code from origin/$BRANCH"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "Building and restarting containers"
docker compose -f deploy/ec2/docker-compose.prod.yml --project-name aura up -d --build --remove-orphans

echo "Pruning dangling images"
docker image prune -f

echo "Deployment complete"
docker compose -f deploy/ec2/docker-compose.prod.yml --project-name aura ps
