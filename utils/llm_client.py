# utils/llm_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import os
import requests


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key, default)
    return v.strip() if isinstance(v, str) else default


def _normalize_base_url(base_url: str) -> str:
    if not base_url:
        return ""
    base_url = base_url.strip()
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    return base_url


def _normalize_model_name(model: str) -> str:
    return model.strip()


def _ensure_openai_completions_endpoint(url_or_base: str) -> str:
    """
    Acepta:
      - base tipo https://host/ollama/v1
      - endpoint final https://host/ollama/v1/chat/completions
    Devuelve endpoint final con /v1/chat/completions.
    """
    u = _normalize_base_url(url_or_base)
    if not u:
        return ""
    # si ya termina en /chat/completions, perfecto
    if u.endswith("/chat/completions"):
        return u
    # si ya incluye /v1 pero no el endpoint, lo añadimos
    if u.endswith("/v1"):
        return f"{u}/chat/completions"
    # si no incluye /v1, asumimos base y añadimos todo
    return f"{u}/v1/chat/completions"


@dataclass
class LLMConfig:
    # Modelo OpenAI-compatible
    model: str = "openai/qwen2.5:32b"
    temperature: float = 0.0
    timeout: float = 120.0

    # Backend: OPENAI u OLLAMA (según su env)
    backend: str = "OPENAI"

    # URLs según env
    llamus_url: str = ""   # .../v1/chat/completions
    ollama_url: str = ""   # .../api/chat
    openai_api_base: str = ""  # .../v1

    # Keys según env
    llamus_api_key: str = ""
    openai_api_key: str = ""


class LLMClient:
    """
    Cliente mínimo compatible con su .env:
    - LLAMUS_BACKEND=OPENAI -> endpoint OpenAI-compatible (/v1/chat/completions)
    - LLAMUS_BACKEND=OLLAMA -> endpoint Ollama (/api/chat)
    """

    def __init__(
        self,
        cfg: Optional[LLMConfig] = None,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        backend: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or LLMConfig()

        # Cargar env (sus variables)
        self.cfg.backend = _env("LLAMUS_BACKEND", self.cfg.backend).upper()
        self.cfg.llamus_url = _env("LLAMUS_URL", self.cfg.llamus_url)
        self.cfg.ollama_url = _env("OLLAMA_URL", self.cfg.ollama_url)
        self.cfg.openai_api_base = _env("OPENAI_API_BASE", self.cfg.openai_api_base)

        self.cfg.llamus_api_key = _env("LLAMUS_API_KEY", self.cfg.llamus_api_key)
        self.cfg.openai_api_key = _env("OPENAI_API_KEY", self.cfg.openai_api_key)

        # Overrides directos
        if model is not None:
            self.cfg.model = model
        else:
            # si no viene override, puede venir del env MODEL_KG_GEN etc.
            self.cfg.model = _env("MODEL_KG_GEN", self.cfg.model)

        if temperature is not None:
            self.cfg.temperature = temperature
        if timeout is not None:
            self.cfg.timeout = timeout
        if backend is not None:
            self.cfg.backend = backend.upper()

        # Resolver endpoint final según backend
        if self.cfg.backend == "OLLAMA":
            # Prioridad: OLLAMA_URL
            self.endpoint = _normalize_base_url(self.cfg.ollama_url)
            if not self.endpoint:
                # fallback razonable si solo han puesto OPENAI_API_BASE
                base = _normalize_base_url(self.cfg.openai_api_base)
                # intentamos transformar /v1 -> /api/chat
                self.endpoint = base.replace("/v1", "/api/chat") if base else ""
            if not self.endpoint:
                raise RuntimeError(
                    "[llm_client] Backend OLLAMA pero no hay OLLAMA_URL ni OPENAI_API_BASE válido."
                )
        else:
            # OPENAI-compatible
            # Prioridad: LLAMUS_URL (ya suele ser /v1/chat/completions)
            if self.cfg.llamus_url:
                self.endpoint = _ensure_openai_completions_endpoint(self.cfg.llamus_url)
            else:
                self.endpoint = _ensure_openai_completions_endpoint(self.cfg.openai_api_base)

            if not self.endpoint:
                raise RuntimeError(
                    "[llm_client] Backend OPENAI pero no hay LLAMUS_URL ni OPENAI_API_BASE válido."
                )

        # Resolver API key
        self.api_key = self.cfg.llamus_api_key or self.cfg.openai_api_key or ""

        if not self.api_key:
            print(
                "[llm_client] Aviso: no hay LLAMUS_API_KEY ni OPENAI_API_KEY definidos."
            )

    # -------- OPENAI BACKEND --------
    def _chat_openai(self, messages: List[Dict[str, str]], temperature: Optional[float]) -> str:
        model_name = _normalize_model_name(self.cfg.model)
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or 'none'}",
        }

        resp = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.cfg.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not isinstance(content, str):
            content = str(content)
        return content

    # -------- OLLAMA BACKEND --------
    def _chat_ollama(self, messages: List[Dict[str, str]], temperature: Optional[float]) -> str:
        model_name = _normalize_model_name(self.cfg.model)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
            },
        }

        headers = {"Content-Type": "application/json"}
        # su adapter puede aceptar bearer también, por si acaso
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.cfg.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Formato Ollama típico:
        # { "message": { "content": "..." }, ... }
        content = data.get("message", {}).get("content", "")
        if not isinstance(content, str):
            content = str(content)
        return content

    # -------- API pública --------
    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        try:
            if self.cfg.backend == "OLLAMA":
                return self._chat_ollama(messages, temperature)
            return self._chat_openai(messages, temperature)
        except Exception as e:
            raise RuntimeError(f"[llm_client] Error en chat ({self.cfg.backend}): {e}")

    def complete(self, *, system: str, user: str, temperature: Optional[float] = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat(messages, temperature=temperature)

    def generate(self, *, input_data: str, context: str, temperature: Optional[float] = None) -> str:
        return self.complete(system=context, user=input_data, temperature=temperature)
