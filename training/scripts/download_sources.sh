#!/usr/bin/env bash
set -euo pipefail

FSC_URL='https://zenodo.org/api/records/11106540/files/fluentai.zip/content'
FSC_ARCHIVE='data/external/fsc/fluentai.zip'
FSC_SHA256='eb7069f505da04e248eb214d53d8c799bcd69f74fe1e4e27d7f61b65e23cdb1f'
GSC_URL='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
GSC_ARCHIVE='data/external/gsc/speech_commands_v0.02.tar.gz'
GSC_SHA256='af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58'

mkdir -p data/external/fsc/raw data/external/gsc/raw

fetch_and_verify() {
  local url=$1
  local output=$2
  local expected_sha256=$3
  if [[ -f "$output" ]] && echo "$expected_sha256  $output" | sha256sum --check --status; then
    echo "Verified existing $output"
    return
  fi
  wget -c -O "$output" "$url"
  echo "$expected_sha256  $output" | sha256sum --check
}

fetch_and_verify "$FSC_URL" "$FSC_ARCHIVE" "$FSC_SHA256"
fetch_and_verify "$GSC_URL" "$GSC_ARCHIVE" "$GSC_SHA256"

if [[ ! -f data/external/fsc/raw/fluent_speech_commands_dataset/data/train_data.csv ]]; then
  unzip -q "$FSC_ARCHIVE" -d data/external/fsc/raw
fi
if [[ ! -f data/external/gsc/raw/validation_list.txt ]]; then
  tar xzf "$GSC_ARCHIVE" -C data/external/gsc/raw
fi

printf '%s  %s\n' "$FSC_SHA256" "$FSC_ARCHIVE" > data/external/fsc/SHA256SUMS
printf '%s  %s\n' "$GSC_SHA256" "$GSC_ARCHIVE" > data/external/gsc/SHA256SUMS

echo "Public source downloads verified and extracted."
echo "Reminder: FSC's bundled license is academic/non-commercial and forbids sharing the dataset/adapted material."
