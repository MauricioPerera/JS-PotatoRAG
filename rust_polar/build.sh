#!/usr/bin/env bash
# Builds rust_polar to WebAssembly and copies the binary to the repo root and
# the PWA public/ dir. The committed rust_polar.wasm must be regenerated with
# this script whenever lib.rs changes.
#
# Requires: rustup target add wasm32-unknown-unknown
set -euo pipefail

cd "$(dirname "$0")"

cargo build --release --target wasm32-unknown-unknown

WASM="target/wasm32-unknown-unknown/release/rust_polar.wasm"
cp "$WASM" ../rust_polar.wasm
cp "$WASM" ../public/rust_polar.wasm

echo "Built and copied rust_polar.wasm ($(wc -c < ../rust_polar.wasm) bytes)"
