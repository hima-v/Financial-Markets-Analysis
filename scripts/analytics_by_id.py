from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request


def build_form(fields: dict[str, str]) -> bytes:
    return urllib.parse.urlencode(fields).encode("utf-8")


def post_form(*, url: str, fields: dict[str, str]) -> dict:
    data = build_form(fields)
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Content-Length", str(len(data)))
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", choices=["movers", "returns", "drawdown", "correlation"])
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--symbol", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--top-n", default="8")
    parser.add_argument("--max-points", default="1500")
    parser.add_argument("--max-symbols", default="30")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    url = f"{args.base_url}/analytics/{args.endpoint}"
    fields: dict[str, str] = {"dataset_id": args.dataset_id}
    if args.start:
        fields["start"] = args.start
    if args.end:
        fields["end"] = args.end

    if args.endpoint == "movers":
        fields["top_n"] = args.top_n
    elif args.endpoint in {"returns", "drawdown"}:
        fields["symbol"] = args.symbol
        fields["max_points"] = args.max_points
    elif args.endpoint == "correlation":
        fields["symbols"] = args.symbols
        fields["max_symbols"] = args.max_symbols

    result = post_form(url=url, fields=fields)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

