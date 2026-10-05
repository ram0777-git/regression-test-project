"""Minimal, robust OpenRouter client (standard library only). Never logs the API key."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

PRICES = {"openai/gpt-5-mini": (0.25, 2.0), "anthropic/claude-sonnet-4.5": (3.0, 15.0)}
RETRY_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def extract_json(text):
    """First decodable JSON object in text (handles ```json fences and prose around it)."""
    if not text:
        return None
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                continue
    return None


class LLM:
    def __init__(self, model=None, fallbacks=None, budget=1.0, deadline=None, log=print, enabled=True):
        first = model or os.getenv("TG_MODEL") or "openai/gpt-5-mini"
        fb = fallbacks if fallbacks is not None else [
            m.strip() for m in (os.getenv("TG_FALLBACK_MODELS") or "").split(",") if m.strip()]
        self.models = [first] + [m for m in fb if m != first]
        self.mi = 0
        base = (os.getenv("TG_LLM_URL") or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        self.budget, self.deadline, self.log = budget, deadline or time.time() + 3600, log
        self.spent, self.calls, self.dead = 0.0, 0, not enabled
        self.timeout = float(os.getenv("TG_LLM_TIMEOUT") or 240)
        self.messages = []

    def available(self) -> bool:
        return (not self.dead and bool(self.key) and self.spent < self.budget * 0.95
                and time.time() < self.deadline - 45)

    def _post(self, messages, max_tokens, temperature):
        body = json.dumps({"model": self.models[self.mi], "messages": messages, "max_tokens": max_tokens,
                           "temperature": temperature, "usage": {"include": True}}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": "Bearer " + self.key, "Content-Type": "application/json",
            "X-Title": "tg-agent"})
        t = max(20.0, min(self.timeout, self.deadline - time.time() - 20))
        with urllib.request.urlopen(req, timeout=t) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def chat(self, messages, max_tokens=16000, temperature=0.2):
        if not self.available():
            return None
        failures = 0
        while failures < 4 and self.available():
            try:
                data = self._post(messages, max_tokens, temperature)
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                self.log("[llm] HTTP %d: %s" % (e.code, detail.replace(self.key, "***") if self.key else detail))
                if e.code == 402:
                    self.dead = True
                    return None
                if e.code in (400, 401, 403, 404) and self.mi + 1 < len(self.models):
                    self.mi += 1
                    self.log("[llm] switching to model %s" % self.models[self.mi])
                    continue
                if e.code in RETRY_CODES:
                    failures += 1
                    time.sleep(min(20, 2 * failures ** 2))
                    continue
                self.dead = True
                return None
            except Exception as e:  # timeouts, DNS, malformed JSON body
                failures += 1
                self.log("[llm] request failed: %s" % type(e).__name__)
                time.sleep(min(10, 2 * failures))
                continue
            if not isinstance(data, dict):
                failures += 1
                continue
            usage = data.get("usage") or {}
            cost = usage.get("cost")
            if not isinstance(cost, (int, float)):
                pin, pout = PRICES.get(self.models[self.mi], (3.0, 15.0))
                cost = ((usage.get("prompt_tokens") or 0) * pin + (usage.get("completion_tokens") or 0) * pout) / 1e6
            self.spent += float(cost)
            self.calls += 1
            try:
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                text = ""
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            self.log("[llm] call %d model=%s cost=$%.4f total=$%.4f" % (
                self.calls, self.models[self.mi], cost, self.spent))
            if text.strip():
                return text
            failures += 1
        if failures >= 4:
            self.log("[llm] giving up on this request")
        return None

    def ask(self, text, max_tokens=16000):
        """Conversation turn (system + first context message kept, middle trimmed)."""
        self.messages.append({"role": "user", "content": text})
        if sum(len(m["content"]) for m in self.messages) > 400000 and len(self.messages) > 8:
            self.messages = self.messages[:2] + [{"role": "assistant", "content": "(earlier rounds omitted)"}] \
                + self.messages[-5:]
            while self.messages[3]["role"] != "user" and len(self.messages) > 4:
                del self.messages[3]
        reply = self.chat(self.messages, max_tokens)
        if reply is None:
            self.messages.pop()
            return None
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def ask_json(self, system, prompt, max_tokens=6000):
        for attempt in range(2):
            text = self.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                             max_tokens, temperature=0.0)
            obj = extract_json(text or "")
            if obj is not None:
                return obj
            if text is None:
                return None
            prompt += "\n\nYour previous answer was not valid JSON. Reply with ONE JSON object only."
        return None
