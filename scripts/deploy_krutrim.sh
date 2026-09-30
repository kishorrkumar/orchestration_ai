#!/usr/bin/env bash
# Symlink or execute root deploy_krutrim.sh
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$DIR/deploy_krutrim.sh" "$@"
