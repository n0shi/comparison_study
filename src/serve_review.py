"""Serve a review page locally, with HTTP Range support so video seeking works.

Python's http.server answers every request with the whole file, and browsers
cannot seek an mp4 served that way (clicking a sentence would restart the clip).

Usage:
    python src/serve_review.py --dir outputs/asr --port 8765
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = _RANGE.match(rng)
        size = os.path.getsize(path)
        if not m:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2) or 0))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)
        self._remaining = None

    def end_headers(self):
        if not self.headers.get("Range"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/asr")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    handler = partial(RangeHandler, directory=a.dir)
    print(f"serving {a.dir} on http://127.0.0.1:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()


if __name__ == "__main__":
    main()
