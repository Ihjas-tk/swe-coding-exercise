"""Minimal HTTP API (stdlib only) that keeps the models warm.

    make serve            # http://127.0.0.1:8080
    curl -s localhost:8080/ask -d '{"question": "..."}'
    GET /health

A fresh `make ask` process pays ~15 s to load the embedding model; this server loads it
once. Responses are the same JSON shape as the CLI, plus latency_ms.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ae import config
from ae.answer.pipeline import Engine
from ae.log import get_logger

log = get_logger(__name__)


def serve(host: str = "127.0.0.1", port: int = 8080, backend: str | None = None) -> None:
    engine = Engine(backend=backend or config.BACKEND)
    engine.ask("warm-up: what is this corpus about?")  # loads the embedding model once
    log.info(f"serving on http://{host}:{port}  (backend={engine.backend})")

    class H(BaseHTTPRequestHandler):
        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health"):
                self._json(200, {"ok": True, "backend": engine.backend})
            else:
                self._json(404, {"error": "use POST /ask"})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/ask"):
                self._json(404, {"error": "use POST /ask"})
                return
            n = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
                q = str(payload.get("question", "")).strip()
                if not q:
                    self._json(400, {"error": "missing 'question'"})
                    return
                a = engine.ask(q, debug=bool(payload.get("debug")))
                out = a.to_json(debug=bool(payload.get("debug")))
                out["latency_ms"] = a.debug.get("timing_ms", {}).get("total")
                self._json(200, out)
            except Exception as e:  # noqa: BLE001
                log.exception("request failed")
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):  # quiet default access log; ours is in ae.log
            log.debug("%s " + fmt, self.address_string(), *args)

    ThreadingHTTPServer((host, port), H).serve_forever()
