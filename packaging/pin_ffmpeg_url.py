"""Turn a BtbN FFmpeg-Builds ".../releases/download/latest/<asset>" URL into the dated autobuild release
that currently holds the same asset, so the licence notice names a download that does not change.

Usage: python3 pin_ffmpeg_url.py URL   (prints the pinned URL, or URL unchanged if it cannot be resolved)
"""

import json
import sys
import urllib.request

LATEST = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"


def pin(url: str) -> str:
    if not url.startswith(LATEST):
        return url
    asset = url[len(LATEST):]
    req = urllib.request.Request("https://api.github.com/repos/BtbN/FFmpeg-Builds/releases?per_page=10",
                                 headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as fh:
            releases = json.load(fh)
    except Exception:  # offline / rate limit: keep the moving URL (the sha256 is still recorded)
        return url
    for rel in releases:
        tag = rel.get("tag_name", "")
        if tag == "latest" or not tag.startswith("autobuild-"):
            continue
        if any(a.get("name") == asset for a in rel.get("assets", [])):
            return f"https://github.com/BtbN/FFmpeg-Builds/releases/download/{tag}/{asset}"
    return url


if __name__ == "__main__":
    print(pin(sys.argv[1]))
