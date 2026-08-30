#!/usr/bin/env bash
# Usage: ./run.sh -m sortbot.main [args]   |   ./run.sh -c "..."
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:$ROOT/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$ROOT/lerobot/.venv/bin/python" "$@"
