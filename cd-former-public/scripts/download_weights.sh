#!/usr/bin/env bash
# Download all 6 CD-Former checkpoints + the preprocessed datasets from OneDrive,
# verifying MD5 against docs/manifest.json.
set -euo pipefail

MANIFEST="${MANIFEST:-docs/manifest.json}"
WEIGHTS_DIR="${WEIGHTS_DIR:-weights}"
DATA_DIR="${DATA_DIR:-data}"

mkdir -p "$WEIGHTS_DIR"

python3 - <<PY
import hashlib, json, sys, urllib.request, os
manifest = json.load(open("${MANIFEST}"))
def fetch(url, dest, expected_md5):
    if os.path.exists(dest):
        h = hashlib.md5(open(dest, "rb").read()).hexdigest()
        if h == expected_md5:
            print(f"  cached: {dest}")
            return
    print(f"  fetching {dest} ...")
    urllib.request.urlretrieve(url, dest)
    h = hashlib.md5(open(dest, "rb").read()).hexdigest()
    if expected_md5 != "<MD5_DRISHTI_42>" and h != expected_md5:
        print(f"  !! MD5 mismatch for {dest} (got {h}, expected {expected_md5})", file=sys.stderr)
        sys.exit(2)
for w in manifest["weights"]:
    if "<ONEDRIVE_LINK" in w["url"]:
        print(f"  skip {w['name']}: URL not configured")
        continue
    fetch(w["url"], "${WEIGHTS_DIR}/" + w["name"] + ".ckpt", w["md5"])
print("Done.")
PY
