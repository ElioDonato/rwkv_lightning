#!/bin/sh
# Invoked by /etc/runit/sv/rwkv-lightning-embed/run via chpst -u donato (mirror
# of service_run.sh) so that whatever PID runsv tracks IS the actual server
# process. Loads the SAME 2.9b model but on a SEPARATE port (8083) so the live
# :8081 chat service stays untouched; this instance exists to serve the
# /embedding and /v1/embeddings endpoints strictly locally.
#   - any RWKV request hitting 8083 also works; /embedding is the intended use.
export HOME=/home/donato
cd /home/donato/rwkv_lightning || exit 1
. ./env.sh
exec /home/donato/rwkv_lightning/.venv/bin/python3 app.py \
  --model-path models/rwkv7-g1i-2.9b-20260805-ctx16384 \
  --port 8083 \
  --password sOLQRwZ5-KVEbPTqO9HCCG9igSHS_j4YXqwzr-m9xx0 \
  >> /home/donato/rwkv_lightning/embedding_8083.log 2>&1