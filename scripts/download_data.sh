#!/usr/bin/env bash
# Download the M5 Forecasting - Accuracy data into data/raw/.
# Option 1 (official): Kaggle CLI. Needs ~/.kaggle/kaggle.json and accepting the
#   competition rules at https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules
# Option 2 (fallback): Nixtla's public mirror of the same files.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
if command -v kaggle >/dev/null 2>&1 && [ -f ~/.kaggle/kaggle.json ]; then
  kaggle competitions download -c m5-forecasting-accuracy -p data/raw
  unzip -o -q data/raw/m5-forecasting-accuracy.zip -d data/raw && rm data/raw/m5-forecasting-accuracy.zip
else
  echo "Kaggle CLI not configured; using the Nixtla mirror"
  curl -sSL -o data/raw/m5.zip https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip
  unzip -o -q data/raw/m5.zip -d data/raw && rm data/raw/m5.zip
fi
ls -lh data/raw
