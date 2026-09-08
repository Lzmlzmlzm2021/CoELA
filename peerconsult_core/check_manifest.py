"""Verify that this vendored copy matches its recorded common-core source."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "CORE_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for name, expected in manifest["sha256"].items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            errors.append(name)
    if errors:
        raise SystemExit("Common-core source mismatch: " + ", ".join(errors))
    print("PeerConsult {}: {} source hashes verified".format(manifest["version"], len(manifest["sha256"])))


if __name__ == "__main__":
    main()
