#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f data/hotel_bookings.csv ]; then
  python3 download_data.py
fi
python3 run.py --data data/hotel_bookings.csv --out results/full
