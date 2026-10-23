"""Download the public source documents into ``data/raw/``.

Files that already exist are skipped. Downloads go to a temporary ``.part``
file first, so an interrupted download never leaves a truncated PDF behind.
"""

import sys
import urllib.request
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# "Medicare & You 2027" (CMS Product No. 10050), linked as "Open PDF" from
# https://www.medicare.gov/publications/search?keywords=10050
DOCUMENTS = {
    "medicare-and-you-2027.pdf": "https://www.medicare.gov/publications/10050-medicare-and-you.pdf",
}

USER_AGENT = "policy-rag-agent/0.1 (+https://github.com/CHIEN-CHOU-WU/policy-rag-agent)"


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        if "pdf" not in content_type:
            raise RuntimeError(f"Expected a PDF from {url}, got Content-Type {content_type!r}")
        partial.write_bytes(response.read())
    partial.replace(destination)


def main() -> int:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in DOCUMENTS.items():
        destination = RAW_DATA_DIR / filename
        if destination.exists():
            print(f"Skipping {filename} (already downloaded)")
            continue
        print(f"Downloading {filename} from {url} ...")
        download(url, destination)
        size_mb = destination.stat().st_size / 1_000_000
        print(f"Saved {destination.relative_to(RAW_DATA_DIR.parent.parent)} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
