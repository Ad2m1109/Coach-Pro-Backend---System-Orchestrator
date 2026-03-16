#!/usr/bin/env bash
set -euo pipefail

# Rotates backend JWT RSA keys used for RS256 signing/verification.
# Creates timestamped backups and writes new keys to backend/certs/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERTS_DIR="$ROOT_DIR/certs"

mkdir -p "$CERTS_DIR"

PRIV="$CERTS_DIR/private.pem"
PUB="$CERTS_DIR/public.pem"

ts="$(date +%Y%m%d%H%M%S)"
if [[ -f "$PRIV" ]]; then
  cp "$PRIV" "$PRIV.bak.$ts"
fi
if [[ -f "$PUB" ]]; then
  cp "$PUB" "$PUB.bak.$ts"
fi

umask 077
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$PRIV"
openssl pkey -in "$PRIV" -pubout -out "$PUB"

chmod 600 "$PRIV" || true
chmod 644 "$PUB" || true

echo "Rotated JWT keys:"
echo "- $PRIV"
echo "- $PUB"
echo "Backups (if any) have suffix .bak.$ts"

