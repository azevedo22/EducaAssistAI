"""
Cliente unificado para LLMs. Suporta três fornecedores intercambiáveis:

  • OLLAMA (local) — default, sem custos, depende de CPU/GPU local.
    Servidor: http://localhost:11434, modelos via `ollama pull`.

  • GROQ (cloud) — tier gratuito, ~200-700 tok/s em hardware LPU.
    Necessita GROQ_API_KEY (https://console.groq.com/keys).
    Limite free tier: 20 000 TPM (8b) / 12 000 TPM (70b).

  • GEMINI (cloud) — Google Gemini via google-genai SDK.
    Necessita GEMINI_LLM_API_KEY (https://aistudio.google.com/apikey).
    Limite free tier: 1 000 000 TPM, 15 RPM, 1 500 req/dia.
    Recomendado quando há rate-limit issues com Groq.

Selecção do fornecedor por env var ``LLM_PROVIDER``:

    LLM_PROVIDER=ollama   → Ollama local (legado)
    LLM_PROVIDER=groq     → Groq cloud
    LLM_PROVIDER=gemini   → Google Gemini cloud  ← recomendado

Variáveis de ambiente reconhecidas:

  Comuns:
    LLM_PROVIDER            "ollama" | "groq" | "gemini"  (default: ollama)

  Ollama:
    OLLAMA_URL              default: http://localhost:11434
    OLLAMA_MODEL            default: gemma2:2b
    OLLAMA_KEEP_ALIVE       default: 15m
    OLLAMA_TIMEOUT          default: 180 (segundos)

  Groq:
    GROQ_API_KEY            obrigatória (sem default)
    GROQ_MODEL              default: llama-3.1-8b-instant
    GROQ_HELPER_MODEL       default: llama-3.1-8b-instant (tarefas auxiliares)
    GROQ_URL                default: https://api.groq.com/openai/v1
    GROQ_TIMEOUT            default: 60 (segundos)

  Gemini:
    GEMINI_LLM_API_KEY      obrigatória (sem default); pode reutilizar GOOGLE_API_KEY
    GEMINI_LLM_MODEL        default: gemini-2.5-flash
    GEMINI_LLM_TIMEOUT      default: 60 (segundos)
"""

import abc
import os
import re as _re
import time as _time
from typing import Any, Dict, List, Optional

import requests


# ----------------------------------------------------------------------
# Exceções partilhadas
# ----------------------------------------------------------------------

class LLMUnavailable(Exception):
    """O fornecedor (Ollama local ou Groq cloud) não está acessível."""


class LLMTimeout(Exception):
    """A geração do LLM excedeu o tempo limite."""


class LLMError(Exception):
    """Outro erro retornado pelo fornecedor (modelo inexistente, body inválido,
    chave API inválida, rate limit, ...)."""


# ----------------------------------------------------------------------
# Interface comum
# ----------------------------------------------------------------------

class BaseLLMClient(abc.ABC):
    """Interface comum a todos os fornecedores. Mantém uma assinatura única
    para que ``answer.py`` e ``app.py`` possam trabalhar com qualquer LLM."""

    provider: str = "base"  # nome do fornecedor (sobrescrito nas subclasses)
    model: str = ""         # modelo configurado (preenchido em __init__)

    @abc.abstractmethod
    def health(self, timeout: int = 5) -> Dict[str, Any]:
        """Devolve um relatório do estado do fornecedor. Não levanta exceções."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        timeout: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        num_predict: Optional[int] = None,
        format_json: bool = False,
        model: Optional[str] = None,
    ) -> str:
        """Geração síncrona. Devolve o texto gerado.

        Levanta ``LLMTimeout``, ``LLMUnavailable`` ou ``LLMError``."""

    def warmup(self, timeout: Optional[int] = None) -> None:
        """Pré-carrega o modelo (se aplicável). Default: pequena chamada para
        forçar o load. Subclasses cloud podem fazer override para no-op."""
        self.generate("Olá.", timeout=timeout, num_predict=8)


# ----------------------------------------------------------------------
# Implementação: Ollama (local)
# ----------------------------------------------------------------------

class OllamaClient(BaseLLMClient):
    """Cliente para um servidor Ollama local (compatível com /api/generate
    e /api/tags). Operações são síncronas/bloqueantes."""

    provider = "ollama"

    def __init__(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        keep_alive: Optional[str] = None,
        default_timeout: Optional[int] = None,
    ):
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma2:2b")
        self.keep_alive = keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", "15m")
        self.default_timeout = (
            default_timeout
            if default_timeout is not None
            else int(os.getenv("OLLAMA_TIMEOUT", "180"))
        )

    def health(self, timeout: int = 5) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=timeout)
            r.raise_for_status()
            tags = r.json()
            available: List[str] = [m.get("name", "") for m in tags.get("models", [])]
            return {
                "ok": True,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "available_models": available,
                "model_available": self.model in available,
                "keep_alive": self.keep_alive,
            }
        except requests.exceptions.ConnectionError as exc:
            return {
                "ok": False,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "error": "ConnectionError",
                "detail": str(exc),
                "hint": (
                    "Ollama não está a correr ou está noutra porta. "
                    "Inicia com `ollama serve` ou verifica OLLAMA_URL."
                ),
            }
        except requests.exceptions.Timeout:
            return {
                "ok": False,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "error": "Timeout",
            }
        except Exception as exc:  # pragma: no cover
            return {
                "ok": False,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "error": type(exc).__name__,
                "detail": str(exc),
            }

    def generate(
        self,
        prompt: str,
        *,
        timeout: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        num_predict: Optional[int] = None,
        format_json: bool = False,
        model: Optional[str] = None,
    ) -> str:
        options: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict

        body: Dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if format_json:
            body["format"] = "json"

        try:
            r = requests.post(
                f"{self.url}/api/generate",
                json=body,
                timeout=timeout if timeout is not None else self.default_timeout,
            )
            r.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise LLMTimeout(
                f"Ollama excedeu {timeout or self.default_timeout}s a responder"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMUnavailable(f"Ollama inacessível em {self.url}") from exc
        except requests.exceptions.HTTPError as exc:
            raise LLMError(f"Erro do Ollama: {exc}") from exc

        try:
            data = r.json()
        except ValueError as exc:
            raise LLMError("Resposta não-JSON do Ollama") from exc

        return (data.get("response") or "").strip()

    def warmup(self, timeout: Optional[int] = None) -> None:
        """Toca no Ollama para forçar o load do modelo em memória.

        Em CPU custa 30–60s na 1ª chamada; depois fica em RAM enquanto
        ``keep_alive`` durar."""
        self.generate("Olá.", timeout=timeout, num_predict=8)


# ----------------------------------------------------------------------
# Implementação: Groq (cloud, tier gratuito)
# ----------------------------------------------------------------------

class GroqClient(BaseLLMClient):
    """Cliente para Groq Cloud — https://groq.com.

    A API é compatível com OpenAI Chat Completions (``/openai/v1/...``).
    O hardware LPU dedicado oferece velocidades de inferência muito
    superiores a CPU local (200–700 tok/s vs. 5–20 tok/s).

    Tier gratuito (à data: maio 2025): tipicamente 30 req/min e ~6000 req/dia.
    Não requer cartão de crédito. Cria a chave em
    https://console.groq.com/keys e exporta como ``GROQ_API_KEY``.
    """

    provider = "groq"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        url: Optional[str] = None,
        default_timeout: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.url = (url or os.getenv("GROQ_URL", "https://api.groq.com/openai/v1")).rstrip("/")
        self.default_timeout = (
            default_timeout
            if default_timeout is not None
            else int(os.getenv("GROQ_TIMEOUT", "60"))
        )

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise LLMError(
                "GROQ_API_KEY não definida. Cria uma chave em "
                "https://console.groq.com/keys e exporta a env var "
                "antes de iniciar o backend."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def health(self, timeout: int = 5) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "ok": False,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "error": "GROQ_API_KEY não definida",
                "hint": "Cria uma chave em https://console.groq.com/keys.",
            }
        try:
            r = requests.get(
                f"{self.url}/models",
                headers=self._headers(),
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()
            available: List[str] = [m.get("id", "") for m in data.get("data", [])]
            return {
                "ok": True,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "available_models": available,
                "model_available": self.model in available,
            }
        except requests.exceptions.ConnectionError as exc:
            return {
                "ok": False,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "error": "ConnectionError",
                "detail": str(exc),
                "hint": "Verifica a tua ligação à internet.",
            }
        except requests.exceptions.Timeout:
            return {
                "ok": False,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "error": "Timeout",
            }
        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                err = r.json().get("error", {})
                detail = err.get("message", "") or str(err)
            except Exception:
                pass
            return {
                "ok": False,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "error": "HTTPError",
                "detail": f"{exc} {detail}".strip(),
                "hint": (
                    "Verifica se a GROQ_API_KEY é válida e não expirou. "
                    "Confirma o modelo em https://console.groq.com/docs/models."
                ),
            }
        except Exception as exc:  # pragma: no cover
            return {
                "ok": False,
                "provider": self.provider,
                "url": self.url,
                "configured_model": self.model,
                "error": type(exc).__name__,
                "detail": str(exc),
            }

    # ── Retry em 429 (rate limit Groq) ──────────────────────────────────
    _MAX_429_RETRIES = 3
    _DEFAULT_429_WAIT = 15.0

    @staticmethod
    def _parse_retry_after(error_msg: str) -> float:
        """Extrai 'try again in X.Xs' da mensagem de erro do Groq."""
        m = _re.search(r"try again in ([\d.]+)s", error_msg or "", _re.IGNORECASE)
        if m:
            return min(float(m.group(1)) + 1.0, 60.0)
        return GroqClient._DEFAULT_429_WAIT

    def generate(
        self,
        prompt: str,
        *,
        timeout: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        num_predict: Optional[int] = None,
        format_json: bool = False,
        model: Optional[str] = None,
    ) -> str:
        body: Dict[str, Any] = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if num_predict is not None:
            body["max_tokens"] = num_predict
        if format_json:
            body["response_format"] = {"type": "json_object"}

        effective_timeout = timeout if timeout is not None else self.default_timeout

        for attempt in range(self._MAX_429_RETRIES + 1):
            try:
                r = requests.post(
                    f"{self.url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                    timeout=effective_timeout,
                )
                r.raise_for_status()
                break  # sucesso — sai do loop
            except requests.exceptions.Timeout as exc:
                raise LLMTimeout(
                    f"Groq excedeu {effective_timeout}s a responder"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                raise LLMUnavailable(f"Groq inacessível em {self.url}") from exc
            except requests.exceptions.HTTPError as exc:
                detail = ""
                try:
                    err = r.json().get("error", {})
                    detail = err.get("message", "") or str(err)
                except Exception:
                    pass

                # 429 rate limit: esperar e re-tentar automaticamente.
                if r.status_code == 429 and attempt < self._MAX_429_RETRIES:
                    wait = self._parse_retry_after(detail)
                    print(
                        f"[groq] 429 rate limit — a aguardar {wait:.1f}s "
                        f"(tentativa {attempt + 1}/{self._MAX_429_RETRIES})"
                    )
                    _time.sleep(wait)
                    continue

                raise LLMError(f"Erro do Groq: {exc}. {detail}".strip()) from exc

        try:
            data = r.json()
        except ValueError as exc:
            raise LLMError("Resposta não-JSON do Groq") from exc

        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()

    def warmup(self, timeout: Optional[int] = None) -> None:
        """Cloud não tem cold-start de modelo, mas validamos a API key e
        a conectividade com um pedido mínimo."""
        try:
            self.generate("Olá.", timeout=timeout, num_predict=8)
        except LLMError as exc:
            # Não bloquear o arranque do backend só porque a key falhou.
            print(f"[warmup] aviso (Groq): {exc}")


# ----------------------------------------------------------------------
# Implementação: Gemini (cloud, google-genai SDK)
# ----------------------------------------------------------------------

class GeminiClient(BaseLLMClient):
    """Cliente para Google Gemini via google-genai SDK.

    Free tier (gemini-2.0-flash): 1 000 000 TPM, 15 RPM, 1 500 req/dia.
    Elimina os rate-limit issues do Groq para prompts grandes.
    Partilha a GEMINI_LLM_API_KEY (distinta da GOOGLE_API_KEY usada para
    embeddings, embora possam ser a mesma chave se o utilizador preferir).
    """

    provider = "gemini"
    _MAX_429_RETRIES = 3

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        default_timeout: Optional[int] = None,
    ):
        # Fallback para GOOGLE_API_KEY se GEMINI_LLM_API_KEY não existir
        self.api_key = (
            api_key
            or os.getenv("GEMINI_LLM_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        ).strip()
        self.model = model or os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
        self.default_timeout = (
            default_timeout
            if default_timeout is not None
            else int(os.getenv("GEMINI_LLM_TIMEOUT", "60"))
        )
        self._genai_client = None   # lazy init (evita import no topo do módulo)

    def _get_sdk_client(self):
        if self._genai_client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise LLMUnavailable(
                    "google-genai não instalado. Corre: pip install google-genai"
                ) from exc
            if not self.api_key:
                raise LLMError(
                    "GEMINI_LLM_API_KEY não definida. Cria uma chave em "
                    "https://aistudio.google.com/apikey e define a env var."
                )
            self._genai_client = genai.Client(api_key=self.api_key)
        return self._genai_client

    def health(self, timeout: int = 5) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "ok": False,
                "provider": self.provider,
                "configured_model": self.model,
                "error": "GEMINI_LLM_API_KEY não definida",
                "hint": "Cria uma chave em https://aistudio.google.com/apikey.",
            }
        try:
            client = self._get_sdk_client()
            # Listar modelos (GET leve, sem consumir tokens) para validar key
            available: List[str] = []
            for m in client.models.list():
                name = getattr(m, "name", "") or ""
                # API devolve "models/gemini-2.5-flash" — normalizar para "gemini-2.5-flash"
                short = name.split("/")[-1] if "/" in name else name
                if short:
                    available.append(short)
            model_available = self.model in available or any(
                self.model in s for s in available
            )
            return {
                "ok": True,
                "provider": self.provider,
                "configured_model": self.model,
                "available_models": available,
                "model_available": model_available,
            }
        except LLMUnavailable as exc:
            return {"ok": False, "provider": self.provider, "configured_model": self.model, "error": str(exc)}
        except Exception as exc:
            return {
                "ok": False,
                "provider": self.provider,
                "configured_model": self.model,
                "error": type(exc).__name__,
                "detail": str(exc),
                "hint": "Verifica se a GEMINI_LLM_API_KEY é válida.",
            }

    def generate(
        self,
        prompt: str,
        *,
        timeout: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        num_predict: Optional[int] = None,
        format_json: bool = False,
        model: Optional[str] = None,
    ) -> str:
        try:
            from google.genai import types as _gt
        except ImportError as exc:
            raise LLMUnavailable("google-genai não instalado") from exc

        effective_model = model or self.model
        thinking_cfg = _gt.ThinkingConfig(thinking_budget=0)
        gen_config = _gt.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=num_predict or 1500,
            thinking_config=thinking_cfg,
            **({"response_mime_type": "application/json"} if format_json else {}),
        )

        sdk_client = self._get_sdk_client()
        last_exc: Optional[Exception] = None

        for attempt in range(self._MAX_429_RETRIES + 1):
            try:
                response = sdk_client.models.generate_content(
                    model=effective_model,
                    contents=prompt,
                    config=gen_config,
                )
                return (response.text or "").strip()
            except Exception as exc:
                msg = str(exc)
                # Rate limit (429) — esperar e re-tentar
                if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                    if attempt < self._MAX_429_RETRIES:
                        wait = self._parse_retry_after(msg)
                        print(
                            f"[gemini] rate limit — a aguardar {wait:.1f}s "
                            f"(tentativa {attempt + 1}/{self._MAX_429_RETRIES})"
                        )
                        _time.sleep(wait)
                        last_exc = exc
                        continue
                # Timeout
                if "timeout" in msg.lower() or "deadline" in msg.lower():
                    raise LLMTimeout(f"Gemini excedeu o tempo limite: {exc}") from exc
                # Sem chave / sem rede
                if "api_key" in msg.lower() or "authentication" in msg.lower():
                    raise LLMError(f"Gemini autenticação falhou: {exc}") from exc
                raise LLMError(f"Erro do Gemini: {exc}") from exc

        raise LLMError(f"Gemini rate limit após {self._MAX_429_RETRIES} tentativas: {last_exc}")

    @staticmethod
    def _parse_retry_after(error_msg: str) -> float:
        m = _re.search(r"retry.{0,20}?(\d+\.?\d*)\s*s", error_msg, _re.IGNORECASE)
        if m:
            return min(float(m.group(1)) + 1.0, 60.0)
        return 15.0

    def warmup(self, timeout: Optional[int] = None) -> None:
        """Valida key e modelo com uma geração mínima."""
        try:
            self.generate("Olá.", num_predict=4)
        except LLMError as exc:
            print(f"[warmup] aviso (Gemini): {exc}")


# ----------------------------------------------------------------------
# Singleton de conveniência
# ----------------------------------------------------------------------

# Alias para compatibilidade retroativa: código antigo que importava
# ``LLMClient`` continua a funcionar e equivale ao cliente Ollama.
LLMClient = OllamaClient

_default_client: Optional[BaseLLMClient] = None


def get_client() -> BaseLLMClient:
    """Devolve a instância singleton do cliente, criando-a com base em
    ``LLM_PROVIDER`` na 1ª chamada."""
    global _default_client
    if _default_client is None:
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        if provider == "groq":
            _default_client = GroqClient()
        elif provider == "gemini":
            _default_client = GeminiClient()
        elif provider in ("ollama", ""):
            _default_client = OllamaClient()
        else:
            raise LLMError(
                f"LLM_PROVIDER desconhecido: '{provider}'. Usa 'ollama', 'groq' ou 'gemini'."
            )
    return _default_client


def reset_client() -> None:
    """Útil para tests ou para forçar a recriação após mudança de env vars."""
    global _default_client
    _default_client = None
