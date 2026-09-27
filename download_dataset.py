"""Download the public-domain Shakespeare dataset from Project Gutenberg."""

from pathlib import Path
from urllib.request import urlopen


URL = "https://www.gutenberg.org/files/100/100-0.txt"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "shakespeare.txt"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from: {URL}")
    with urlopen(URL, timeout=30) as response:
        content = response.read().decode("utf-8")
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Saved dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
