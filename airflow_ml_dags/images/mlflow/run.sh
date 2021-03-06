#!/bin/sh

set -e

if [ -z "$FILE_DIR" ]; then
  echo >&2 "FILE_DIR must be set"
  exit 1
fi

mkdir -p "$FILE_DIR" && mlflow server \
    --backend-store-uri sqlite:///${FILE_DIR}/sqlite.db \
    --host 0.0.0.0 \
    --port $PORT