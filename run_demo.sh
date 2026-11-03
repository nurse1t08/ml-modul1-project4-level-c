#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 generate_demo.py
python3 run.py --data data/hotel_bookings_demo.csv --out results/demo
