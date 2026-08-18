#!/bin/sh
# Generic RWKV Lightning server launcher.
#
# Launches the RWKV Lightning HTTP server (app.py). Every knob is driven by an
# environment variable; nothing here is machine-, project-, or
# credential-specific, so the same script runs from any checkout of this repo,
# under any service manager (runit, systemd, a shell, ...).
#
#   RWKV_MODEL_PATH       Directory of model weights. REQUIRED for a service to
#                         start; app.py errors out clearly when it is empty.
#   RWKV_INFERENCE_ENGINE Model backend: fp16 | GemLite | CUTLASS (default fp16).
#   RWKV_PORT             Port to bind (default 8000). Co-resident services MUST
#                         use DISTINCT RWKV_PORT values.
#   RWKV_HOST             Bind host (default 0.0.0.0).
#   RWKV_API_PASSWORD     API auth. Auth is configured via this env var (and,
#                         on the app.py side, RWKV_API_PASSWORD in settings.py);
#                         the password must NEVER be committed to the repo.
#   RWKV_LOG_FILE         If set, stdout+stderr are redirected here; otherwise
#                         they are left to whatever runs the service.
#   RWKV_REPO_DIR         Repo root; defaults to this script's own directory.
#   RWKV_VENV_PYTHON      Python interpreter; defaults to
#                         $REPO_DIR/.venv/bin/python3 if present, else python3.
#   RWKV_ENV_FILE         Optional local env file to source (defaults to
#                         $REPO_DIR/env.sh). Only sourced if it exists.
#
# Intent: this instance serves the /embedding and /v1/embeddings endpoints.
# The logic is shared with service_run.sh (the chat server); they differ only
# in intent, so make any change to both.

REPO_DIR="${RWKV_REPO_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}"

# Optionally source a local env file. runit may not provide the login env, so
# keep this if you deploy one (env.sh is an example). Required length:
# it must export RWKV_MODEL_PATH before the exec below.
ENV_FILE="${RWKV_ENV_FILE:-$REPO_DIR/env.sh}"
if [ -f "$ENV_FILE" ]; then
    . "$ENV_FILE"
fi

# Pick a Python interpreter: explicit override, else repo venv, else system.
if [ -n "${RWKV_VENV_PYTHON:-}" ]; then
    PYTHON_BIN="$RWKV_VENV_PYTHON"
elif [ -x "$REPO_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python3"
else
    PYTHON_BIN=python3
fi

cd "$REPO_DIR" || exit 1

# --model-path is required; pass it through so app.py's own check fires if
# RWKV_MODEL_PATH is not set. Append optional overrides only when configured.
set -- app.py --model-path "${RWKV_MODEL_PATH:-}"
[ -n "${RWKV_INFERENCE_ENGINE:-}" ] && set -- "$@" --inference-engine "$RWKV_INFERENCE_ENGINE"
[ -n "${RWKV_PORT:-}" ] && set -- "$@" --port "$RWKV_PORT"

# Auth is read automatically by settings.py from the RWKV_API_PASSWORD env var;
# do not pass --password here, and never commit a password value.

# Exec (not fork) so the PID the service manager tracks IS the server process.
if [ -n "${RWKV_LOG_FILE:-}" ]; then
    exec "$PYTHON_BIN" "$@" >> "$RWKV_LOG_FILE" 2>&1
else
    exec "$PYTHON_BIN" "$@"
fi