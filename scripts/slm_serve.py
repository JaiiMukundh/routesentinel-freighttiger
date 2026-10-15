"""Tiny local-SLM shim: Ollama-compatible /api/chat backed by transformers on CPU.

Deviation from AGENTS.md (user-approved, recorded in NOTES.md):
- Real Ollama binary needs ~1.8GB + root; this sandbox has neither. Instead a
  stdlib HTTP server mimics Ollama's /api/chat + /api/tags on localhost:11434
  and runs Qwen2.5-0.5B-Instruct locally via torch/transformers (greedy,
  seed 42). No hosted/paid API is ever touched; `ollama.chat()` works unchanged.
- config.yaml keeps the canonical `qwen2.5:3b-instruct-q4_K_M`; pass
  --model to serve a smaller demo model. Verdicts are unaffected (LLM never
  decides; verifier + template fallback still apply).

Usage:
  python3 scripts/slm_serve.py --model-dir /tmp/opencode/models/qwen05 [--port 11434]
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "qwen2.5:0.5b-instruct"

_pipe_lock = threading.Lock()
_pipe = None
_loaded_name = "qwen2.5:0.5b"


def load_pipe(model_dir: str):  # type: ignore[no-untyped-def]
    """Load tokenizer + model once (CPU, eval mode, greedy decoding)."""
    global _pipe, _loaded_name
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    model.eval()
    _pipe = (tok, model)
    _loaded_name = "qwen2.5:0.5b"
    return _pipe


def generate_text(prompt: str, max_new_tokens: int = 256) -> str:
    """Greedy local generation (temperature=0 equivalent), seeded."""
    import torch

    assert _pipe is not None, "model not loaded"
    tok, model = _pipe
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt")
    with torch.no_grad(), _pipe_lock:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "slm-shim/0.1"

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/tags":
            self._json({"models": [{"name": _loaded_name, "model": _loaded_name}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json({"error": "bad json"}, 400)
            return
        messages = req.get("messages", [])
        prompt = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        if not prompt.strip():
            self._json({"error": "empty prompt"}, 400)
            return
        try:
            content = generate_text(prompt)
        except Exception as exc:  # noqa: BLE001 — surfaced as 500, client retries/falls back
            self._json({"error": f"generation failed: {exc}"}, 500)
            return
        self._json(
            {
                "model": req.get("model", _loaded_name),
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
            }
        )

    def log_message(self, *args: object) -> None:
        pass  # quiet; use --verbose to enable


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local SLM shim (Ollama-compatible).")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    load_pipe(args.model_dir)
    if args.verbose:
        Handler.log_message = BaseHTTPRequestHandler.log_message  # type: ignore[method-assign]
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"slm-shim serving {_loaded_name} on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
