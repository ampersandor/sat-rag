"""Fetch reference manuals and store them as plain text files."""
from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SOURCES = {
    "mafft_manual.txt": "https://mafft.cbrc.jp/alignment/software/manual/manual.html",
    "usearch_uclust_manual.txt": "https://drive5.com/usearch/manual/cmds_all.html",
}


def clean_text(text: str) -> str:
    """Collapse repeated whitespace and normalise spacing."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" (?=[,.;:!?])", "", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()



def extract_plain_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    filtered = [line for line in lines if line]
    return clean_text("\n".join(filtered))



def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        text = extract_plain_text(resp.text)
        (DATA_DIR / filename).write_text(text, encoding="utf-8")
        print(f"Saved {filename} from {url} ({len(text)} chars)")


if __name__ == "__main__":
    main()
