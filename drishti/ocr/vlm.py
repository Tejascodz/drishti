"""Vision-language model reader: a second opinion for hard plates, and a labelling aid.

Two backends:
  ollama   Ollama's native /api/chat (images as base64, thinking off, JSON-schema
           output). Default; runs fully on a laptop, e.g. `ollama pull qwen3-vl:2b`.
  openai   any OpenAI-compatible server (llama.cpp, vLLM, LM Studio, hosted APIs).

At recognition time the VLM never gets the last word on its own: it sees the crop
and the decoder's ranked candidates, and its answer is accepted only if it is a
valid plate under the active format grammar AND either one of the candidates or
within one edit of the top read. For labelling, `read()` asks without candidates
so the VLM gives an independent second reading.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from ..formats.grammar import FormatGrammar
from ..fusion import levenshtein

log = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {"plate": {"type": "string"}, "readable": {"type": "boolean"}},
    "required": ["plate", "readable"],
}

_READ_PROMPT = (
    "This image is a cropped vehicle registration plate{region}. Read every character on the plate. "
    "For a two-line plate read the top line first, then the bottom line. Ignore emblems, the 'IND' mark, "
    "dealer names and stickers. Use only A-Z and 0-9, no spaces. "
    'Reply as JSON: {{"plate": "...", "readable": true}}. If the characters cannot be read, '
    'reply {{"plate": "", "readable": false}}.'
)
_CHOOSE_SUFFIX = " An OCR system suggests these readings, most likely first: {cands}. They may all be wrong."

_REGION_HINT = {
    "india": " from India (formats like KA05MX3021, DL1CR9541, MH12A1234, 21BH1234AB)",
    "brazil": " from Brazil (formats like ABC1234 or ABC1D23)",
}


@dataclass
class VLMAnswer:
    text: str | None
    readable: bool
    raw: str = ""


def _clean(s: str) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


class VLMVerifier:
    def __init__(self, endpoint: str, model: str, api_key_env: str = "DRISHTI_VLM_KEY", timeout_s: float = 30.0,
                 backend: str = "ollama", region: str | None = None, keep_alive: str = "10m"):
        self.backend = backend.lower()
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout_s
        self.keep_alive = keep_alive
        self.region_hint = _REGION_HINT.get(str(region or "").lower(), "")
        self._send_think = True

    # --------------------------------------------------------------- transport
    def _post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.endpoint + path, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    @staticmethod
    def _encode(crop: np.ndarray) -> str:
        h, w = crop.shape[:2]
        if h < 96:  # small VLMs read better from a moderately upscaled crop
            s = 96 / h
            crop = cv2.resize(crop, (int(round(w * s)), 96), interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            raise ValueError("could not encode crop")
        return base64.b64encode(buf.tobytes()).decode()

    def _chat(self, prompt: str, image_b64: str) -> str:
        if self.backend == "ollama":
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
                "stream": False,
                "format": _SCHEMA,
                "options": {"temperature": 0, "num_predict": 64},
                "keep_alive": self.keep_alive,
            }
            if self._send_think:
                body["think"] = False
            try:
                data = self._post("/api/chat", body)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="ignore") if hasattr(exc, "read") else ""
                if self._send_think and "think" in detail.lower():
                    self._send_think = False  # model without a thinking switch
                    body.pop("think", None)
                    data = self._post("/api/chat", body)
                else:
                    raise
            return str(data.get("message", {}).get("content", ""))
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 64,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
            ]}],
        }
        data = self._post("/chat/completions", body)
        return str(data["choices"][0]["message"]["content"])

    @staticmethod
    def _parse(content: str) -> VLMAnswer:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            txt = _clean(content)
            return VLMAnswer(txt or None, bool(txt), content)
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return VLMAnswer(None, False, content)
        txt = _clean(d.get("plate", ""))
        readable = bool(d.get("readable", bool(txt))) and bool(txt)
        return VLMAnswer(txt if readable else None, readable, content)

    # ------------------------------------------------------------------- API
    def read(self, crop: np.ndarray, candidates: list[str] | None = None) -> VLMAnswer:
        """Read a plate crop. Raises on transport errors (callers decide how to degrade)."""
        prompt = _READ_PROMPT.format(region=self.region_hint)
        if candidates:
            prompt += _CHOOSE_SUFFIX.format(cands=", ".join(candidates[:5]))
        return self._parse(self._chat(prompt, self._encode(crop)))

    def verify(self, crop: np.ndarray, top_text: str, alternatives: list[tuple[str, float]], grammar: FormatGrammar) -> str | None:
        cands = [t for t, _ in alternatives] or [top_text]
        try:
            ans = self.read(crop, cands)
        except Exception as exc:  # network / server errors must never break recognition
            log.warning("VLM verification skipped: %s", exc)
            return None
        if not ans.text or grammar.is_valid(ans.text) is None:
            return None
        if ans.text in cands or levenshtein(ans.text, top_text) <= 1:
            return ans.text
        return None

    def check(self) -> dict:
        """Reachability and model availability, for `drishti vlm-check`."""
        info = {"backend": self.backend, "endpoint": self.endpoint, "model": self.model}
        try:
            if self.backend == "ollama":
                with urllib.request.urlopen(self.endpoint + "/api/tags", timeout=5) as resp:
                    names = [m.get("name", "") for m in json.loads(resp.read().decode()).get("models", [])]
                info["reachable"] = True
                info["installed_models"] = names
                want = self.model if ":" in self.model else self.model + ":latest"
                info["model_present"] = want in names or self.model in names
                if not info["model_present"]:
                    info["hint"] = f"ollama pull {self.model}"
            else:
                req = urllib.request.Request(self.endpoint + "/models",
                                             headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    names = [m.get("id", "") for m in json.loads(resp.read().decode()).get("data", [])]
                info["reachable"] = True
                info["installed_models"] = names
                info["model_present"] = self.model in names or not names
        except Exception as exc:
            info["reachable"] = False
            info["error"] = str(exc)
            if self.backend == "ollama":
                info["hint"] = "start Ollama (open the app or run `ollama serve`)"
        return info


def from_config(vcfg: dict, region: str | None = None) -> VLMVerifier:
    return VLMVerifier(
        endpoint=vcfg.get("endpoint", "http://localhost:11434"),
        model=vcfg.get("model", "qwen3-vl:2b"),
        api_key_env=vcfg.get("api_key_env", "DRISHTI_VLM_KEY"),
        timeout_s=float(vcfg.get("timeout_s", 30.0)),
        backend=vcfg.get("backend", "ollama"),
        region=region,
        keep_alive=str(vcfg.get("keep_alive", "10m")),
    )
