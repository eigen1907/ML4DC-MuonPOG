#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --locked python scripts/fetch.py \
  --dataset '/Muon0/Run2025D-PromptReco-v1/DQMIO' \
  --run 395719 \
  --me 'Muons/MuonRecoAnalyzer/GlbMuon_Glb_eta'

uv run --locked python scripts/plot.py \
  --input 'data/raw/Muon0__Run2025D-PromptReco-v1__DQMIO/395719/Muons__MuonRecoAnalyzer__GlbMuon_Glb_eta.parquet'
