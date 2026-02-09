from __future__ import annotations

import argparse
import json
import pathlib
import uuid
import urllib.request


def build_multipart(*, field_name: str, filename: str, content_type: str, payload: bytes) -> tuple[str, bytes]:
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return boundary, head + payload + tail


def post_file(*, url: str, path: pathlib.Path) -> dict:
    payload = path.read_bytes()
    boundary, body = build_multipart(field_name="file", filename=path.name, content_type="text/csv", payload=payload)
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/datasets/upload")
    parser.add_argument("--path", default="nse_sensex (1).csv")
    args = parser.parse_args()

    result = post_file(url=args.url, path=pathlib.Path(args.path))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

