"""TradingView's legal integration path: your own Pine Script alert posts JSON to
this endpoint, and the bot records it as a signal. Nothing is scraped from
TradingView and nothing here places orders.

Pine alert message example:
  {"secret": "{{your-shared-secret}}", "ticker": "{{ticker}}", "action": "buy", "price": {{close}}}

Run:  STOCKSBOT_WEBHOOK_SECRET=... python -m stocksbot.webhook --port 8080
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "results" / "tv_signals.jsonl"
ACTIONS = {"buy", "sell", "flat"}


def parse_alert(body: bytes, secret: str) -> dict:
    msg = json.loads(body)
    if not hmac.compare_digest(str(msg.get("secret", "")), secret):
        raise PermissionError("bad secret")
    if msg.get("action") not in ACTIONS or not msg.get("ticker"):
        raise ValueError("alert needs ticker and action in buy/sell/flat")
    return {"ts": datetime.now(timezone.utc).isoformat(), "ticker": msg["ticker"],
            "action": msg["action"], "price": msg.get("price")}


class Handler(BaseHTTPRequestHandler):
    secret = ""

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            alert = parse_alert(body, self.secret)
        except PermissionError:
            self.send_response(403); self.end_headers(); return
        except (ValueError, json.JSONDecodeError):
            self.send_response(400); self.end_headers(); return
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(alert) + "\n")
        self.send_response(200); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    Handler.secret = os.environ["STOCKSBOT_WEBHOOK_SECRET"]
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
