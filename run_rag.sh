#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

python scripts/rag_inference_config.py --config config/rag_config.json
