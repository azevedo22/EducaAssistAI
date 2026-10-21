"""
Retrieval e abstração de provider de embeddings.

Suporta dois fornecedores intercambiáveis para gerar embeddings:

  • ``local`` (default) — sentence-transformers em CPU.
    Modelo: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, ~471 MB).
    Sem custos, sem internet, mais lento (~25–60 chunks/s).

  • ``gemini`` — Google Generative AI Embeddings (cloud).
    Modelo: text-embedding-004 (768-dim).
    Tier gratuito: 1500 req/min, sem cap diário.
    Necessita ``GOOGLE_API_KEY``.

Selecção do provider por env var ``EMBED_PROVIDER``:

    EMBED_PROVIDER=local    → sentence-transformers (default)
    EMBED_PROVIDER=gemini   → Google Gemini Embeddings

IMPORTANTE: cada provider gera vetores num espaço semântico diferente
(dimensões e distribuições incompatíveis). Por isso usamos uma coleção
ChromaDB POR PROVIDER — ``collection_name_for_provider()`` produz o nome
correto. Trocar de provider implica re-carregar os PDFs nesse modo
(ou seja, o índice local é preservado quando se passa para Gemini, e
vice-versa, permitindo comparações lado-a-lado para o relatório).
"""

import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import requests
from chromadb.utils import embedding_functions


# ---- Modelos por defeito (sobrescrevíveis via env vars) -------------------

# Modelo multilingue (PT/EN/ES/...): mesma dimensão (384) e tamanho que o
# MiniLM original, mas treinado em ~50 línguas. Embebe corretamente termos
# como "sexta-feira", "gabinete", "horário" — críticos para este corpus.
LOCAL_EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Gemini text-embedding-004 — 768 dim, suporte multilingue de topo
# (https://ai.google.dev/gemini-api/docs/embeddings).
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")


# ---- Selecção de provider --------------------------------------------------

def get_embed_provider() -> str:
    """Devolve o nome do provider ativo, normalizado."""
    return os.getenv("EMBED_PROVIDER", "local").strip().lower() or "local"


def collection_name_for_provider(base_name: str = "ipb_docs_v3") -> str:
    """Nome de coleção único por provider — evita misturar vetores
    incompatíveis na mesma coleção. Permite manter um índice local e
    um índice cloud em paralelo, alternando via env var."""
    return f"{base_name}_{get_embed_provider()}"


def list_gemini_models(api_key: str, timeout: int = 15) -> Dict[str, Any]:
    """Lista modelos Gemini disponíveis para a chave fornecida e separa
    os que suportam embeddings dos que suportam geração. Útil para
    diagnóstico quando uma chamada falha com 404 'model not found'.
    """
    if not api_key:
        return {"ok": False, "error": "GOOGLE_API_KEY não definida"}
    try:
        r = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
            timeout=timeout,
        )
        r.raise_for_status()
    except requests.exceptions.RequestException as exc:
        detail = ""
        try:
            detail = r.json().get("error", {}).get("message", "")
        except Exception:
            pass
        return {
            "ok": False,
            "error": str(exc),
            "detail": detail,
        }

    data = r.json() or {}
    models = data.get("models", []) or []
    embedding_models: List[Dict[str, Any]] = []
    generation_models: List[Dict[str, Any]] = []
    for m in models:
        methods = set(m.get("supportedGenerationMethods", []) or [])
        info = {
            "name": m.get("name"),
            "displayName": m.get("displayName"),
            "supportedGenerationMethods": sorted(methods),
        }
        if {"embedContent", "batchEmbedContents"} & methods:
            embedding_models.append(info)
        if "generateContent" in methods:
            generation_models.append(info)

    return {
        "ok": True,
        "embedding_models": embedding_models,
        "generation_models": generation_models,
    }


class _GeminiEmbeddingFunction:
    """Embedding function customizada para a Gemini API.

    Razão de existir: a ``GoogleGenerativeAiEmbeddingFunction`` do ChromaDB
    depende internamente do pacote ``google-generativeai`` (oficialmente
    deprecated em 2025), que tem incompatibilidades de versões com o
    ``google-api-core`` mais recente. Em vez de pinar versões frágeis,
    falamos diretamente com a API REST via ``requests`` (que já era
    dependência do projeto).

    Cumpre o protocolo ``EmbeddingFunction`` do ChromaDB:
    ``__call__(input) -> List[List[float]]``.

    Estratégia de robustez: tenta ``:batchEmbedContents`` primeiro (1 call
    por lote, mais rápido). Se a API responder 404 ("model not found",
    típico em modelos novos não totalmente lançados em todas as regiões),
    cai para ``:embedContent`` (chamada por documento, mais lento mas
    universalmente suportado). Se ambos falharem, devolve uma mensagem
    com os modelos que a chave tem acesso, para o utilizador escolher.
    """

    # Tier gratuito: 100 req/min para embeddings. Limitamos a sub-batches
    # pequenos para que cada upload de PDF não esgote a quota num pedido só.
    DEFAULT_FREE_TIER_BATCH = 25
    MAX_429_RETRIES = 6
    DEFAULT_RETRY_WAIT = 30.0

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/text-embedding-004",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 60,
        batch_size: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("api_key é obrigatória para Gemini Embeddings.")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Batch sub-divide o input em pequenos lotes para gerir a quota.
        # Pode ser sobrescrito via env var ``GEMINI_EMBED_BATCH``.
        self.batch_size = batch_size or int(
            os.getenv("GEMINI_EMBED_BATCH", str(self.DEFAULT_FREE_TIER_BATCH))
        )
        # Memoriza qual endpoint funciona (batch vs single). Valores:
        # None (não testado), "batch", "single".
        self._working_endpoint: Optional[str] = None

    # ---- helpers HTTP ------------------------------------------------------

    def _hint_with_models(self) -> str:
        """Procura modelos de embeddings disponíveis para a chave e devolve
        uma mensagem que ajuda o utilizador a escolher um nome válido."""
        info = list_gemini_models(self.api_key, timeout=10)
        if not info.get("ok"):
            return ""
        names = [m.get("name", "") for m in info.get("embedding_models", [])]
        if not names:
            return (
                "  Nenhum modelo de embeddings disponível para esta chave. "
                "Verifica em https://aistudio.google.com/apikey."
            )
        return (
            "  Modelos de embeddings disponíveis para a tua chave:\n"
            + "\n".join(f"    - {n}" for n in names)
            + f"\n  Define GEMINI_EMBED_MODEL para um destes (atual: {self.model_name})."
        )

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> float:
        """Determina quanto tempo esperar antes de re-tentar um 429.

        Lê primeiro o header ``Retry-After`` (segundos ou data HTTP);
        se ausente, procura ``"retry in 16.53s"`` na mensagem do JSON
        (formato típico do Google). Cai para um default razoável."""
        # 1) Header padrão HTTP
        ra = response.headers.get("Retry-After", "").strip()
        if ra:
            try:
                return float(ra) + 0.5
            except ValueError:
                pass
        # 2) Mensagem dentro do JSON
        try:
            msg = (response.json() or {}).get("error", {}).get("message", "") or ""
        except Exception:
            msg = ""
        m = re.search(r"retry\s+in\s+([\d.]+)\s*s", msg, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1)) + 0.5
            except ValueError:
                pass
        return _GeminiEmbeddingFunction.DEFAULT_RETRY_WAIT

    def _post(self, url: str, payload: Dict[str, Any]) -> requests.Response:
        """POST com retry automático em 429 (rate limit). Outros erros HTTP
        propagam-se imediatamente. Loga as esperas para a consola para o
        utilizador ter visibilidade do que está a acontecer."""
        last_response: Optional[requests.Response] = None
        for attempt in range(1, self.MAX_429_RETRIES + 1):
            r = requests.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
            last_response = r
            if r.status_code != 429:
                r.raise_for_status()
                return r

            # 429: respeitar a hint de retry da Google.
            wait = self._parse_retry_after(r)
            if attempt >= self.MAX_429_RETRIES:
                break
            print(
                f"[gemini embed] 429 quota — a esperar {wait:.1f}s "
                f"antes de re-tentar (tentativa {attempt}/{self.MAX_429_RETRIES})"
            )
            time.sleep(wait)

        # Esgotámos retries; lançar o erro do último 429.
        assert last_response is not None
        last_response.raise_for_status()
        return last_response  # pragma: no cover

    def _post_batch(self, batch: List[str]) -> List[List[float]]:
        payload = {
            "requests": [
                {
                    "model": self.model_name,
                    "content": {"parts": [{"text": text}]},
                }
                for text in batch
            ]
        }
        r = self._post(
            f"{self.base_url}/{self.model_name}:batchEmbedContents", payload
        )
        data = r.json() or {}
        return [(emb.get("values") or []) for emb in (data.get("embeddings") or [])]

    def _post_single(self, text: str) -> List[float]:
        payload = {
            "model": self.model_name,
            "content": {"parts": [{"text": text}]},
        }
        r = self._post(
            f"{self.base_url}/{self.model_name}:embedContent", payload
        )
        data = r.json() or {}
        return (data.get("embedding") or {}).get("values") or []

    # ---- contrato ChromaDB -------------------------------------------------

    @staticmethod
    def _coerce_to_list(value: Any) -> List[str]:
        """Aceita str, lista, tuple, ou qualquer iterável e devolve
        sempre uma ``List[str]``. Útil porque diferentes versões do
        ChromaDB chamam-nos com convenções diferentes."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(x) for x in value]
        try:
            return [str(x) for x in value]
        except TypeError:
            return [str(value)]

    def __call__(self, input):
        # ASSINATURA ESTRITA exigida pelo ChromaDB:
        # ``def __call__(self, input)`` — não usar *args/**kwargs aqui.
        # O parâmetro pode ser str ou list; coercimos lá dentro.
        items = self._coerce_to_list(input)
        if not items:
            return []
        # Reatribuir ao nome local 'input' para o resto do método continuar
        # a usar a variável que tinha antes da refactorização.
        input = items

        # Se já sabemos que só funciona em 'single', usá-lo direto.
        if self._working_endpoint == "single":
            return [self._post_single(t) for t in input]

        results: List[List[float]] = []
        for start in range(0, len(input), self.batch_size):
            batch = input[start:start + self.batch_size]

            # 1ª tentativa: batch (mais rápido).
            try:
                batch_results = self._post_batch(batch)
                self._working_endpoint = "batch"
                results.extend(batch_results)
                continue
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                # Só fallback para single em 400/404 (modelo não suporta batch).
                # Para 401/403/429, propagar — não vale a pena tentar 'single'.
                if status not in (400, 404):
                    detail = ""
                    try:
                        detail = exc.response.json().get("error", {}).get("message", "")
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Falha a obter embeddings Gemini ({status}): {exc}. "
                        f"{detail}".strip()
                    ) from exc

            # 2ª tentativa: chamada singular.
            try:
                singular_results = [self._post_single(t) for t in batch]
                self._working_endpoint = "single"
                results.extend(singular_results)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                detail = ""
                try:
                    detail = exc.response.json().get("error", {}).get("message", "")
                except Exception:
                    pass
                hint = ""
                if status in (400, 404):
                    hint = "\n" + self._hint_with_models()
                raise RuntimeError(
                    f"Falha a obter embeddings Gemini ({status}): {exc}. "
                    f"{detail}{hint}".rstrip()
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(
                    f"Falha de rede a obter embeddings Gemini: {exc}"
                ) from exc

        return results

    # ---- contrato ChromaDB para embed_query / embed_documents -----------
    # NOTA IMPORTANTE: ao contrário da convenção langchain, ChromaDB usa o
    # retorno destes métodos diretamente como ``query_embeddings`` —
    # logo, ambos têm de devolver ``List[List[float]]`` (uma lista de
    # embeddings), não ``List[float]``. Devolver um vetor plano fá-lo
    # falhar com ``argument 'query_embeddings': 'float' object cannot be
    # converted to 'Sequence'`` no nível do Rust binding.

    def embed_documents(self, input):
        """Embeddings para múltiplos documentos. Retorno: ``List[List[float]]``."""
        return self.__call__(input)

    def embed_query(self, input):
        """Embedding para uma query única (1 texto). Retorno: ``List[List[float]]``
        com um único embedding lá dentro, para que ChromaDB possa usá-lo
        como ``query_embeddings`` sem re-wrapping."""
        items = self._coerce_to_list(input)
        if not items:
            return []
        # Devolver no formato Embeddings (lista de listas), não List[float],
        # mesmo para uma única query — é o que ChromaDB espera.
        return self.__call__([items[0]])

    def name(self) -> str:
        return f"gemini-{self.model_name.replace('/', '-')}"


def get_embedding_function():
    """Constrói a embedding function correta para o provider configurado.

    Erros: se Gemini estiver selecionado e ``GOOGLE_API_KEY`` não existir,
    levanta ``RuntimeError`` com instruções claras (em vez de falhar
    obscuramente lá dentro).
    """
    provider = get_embed_provider()

    if provider == "local":
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=LOCAL_EMBED_MODEL,
        )

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY não definida. Cria uma chave em "
                "https://aistudio.google.com/apikey e exporta a env var "
                "antes de iniciar o backend (ou volta a EMBED_PROVIDER=local)."
            )
        return _GeminiEmbeddingFunction(
            api_key=api_key,
            model_name=GEMINI_EMBED_MODEL,
        )

    raise RuntimeError(
        f"EMBED_PROVIDER desconhecido: '{provider}'. Usa 'local' ou 'gemini'."
    )


# ---- Health helper ---------------------------------------------------------

def embed_health() -> Dict[str, Any]:
    """Devolve o estado do provider de embeddings, sem levantar exceções.

    Para o frontend mostrar um badge e para diagnóstico via curl. NÃO faz
    chamadas de teste à cloud (evita gastar quota); apenas reporta a
    configuração e se as credenciais necessárias estão presentes."""
    provider = get_embed_provider()
    if provider == "local":
        return {
            "ok": True,
            "provider": "local",
            "configured_model": LOCAL_EMBED_MODEL,
            "dimensions": 384,
            "note": "Embeddings calculados em CPU via sentence-transformers.",
        }

    if provider == "gemini":
        has_key = bool(os.getenv("GOOGLE_API_KEY", "").strip())
        return {
            "ok": has_key,
            "provider": "gemini",
            "configured_model": GEMINI_EMBED_MODEL,
            "dimensions": 768,
            "error": None if has_key else "GOOGLE_API_KEY não definida",
            "hint": (
                None
                if has_key
                else "Cria uma chave em https://aistudio.google.com/apikey."
            ),
        }

    return {
        "ok": False,
        "provider": provider,
        "error": f"Provider desconhecido: {provider}",
    }


# ---- Coleção ChromaDB ------------------------------------------------------

def create_collection(persist_directory: str = "backend/db", name: Optional[str] = None):
    """Cria (ou abre) uma coleção ChromaDB com a embedding function do
    provider ativo. Se ``name`` não for passado, usa o nome derivado do
    provider, garantindo isolamento entre `local` e `gemini`.

    Idempotente: usa ``get_or_create_collection`` para evitar a race
    condition entre "coleção não existe" e "InternalError: já existe"
    em versões recentes do ChromaDB. O ``try``/``except`` serve apenas
    para versões antigas que não tenham ``get_or_create_collection``.
    """
    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)

    if name is None:
        name = collection_name_for_provider()

    embed_fn = get_embedding_function()

    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(
            name=name, embedding_function=embed_fn
        )

    # Fallback para versões muito antigas (pré-0.4) que não têm
    # get_or_create_collection. Tenta primeiro abrir; se realmente não
    # existir (e só nesse caso), cria.
    try:
        return client.get_collection(name=name, embedding_function=embed_fn)
    except (ValueError, KeyError):
        return client.create_collection(name=name, embedding_function=embed_fn)


def index_documents(collection, texts: List[str], metadatas: List[Dict[str, str]], ids: List[str]):
    if not texts or not ids:
        return

    collection.add(documents=texts, metadatas=metadatas, ids=ids)


def query_collection(collection, question: str, k: int = 3) -> List[Dict[str, str]]:
    if not question.strip():
        return []

    results = collection.query(query_texts=[question], n_results=k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return [
        {
            "text": documents[idx] if idx < len(documents) else "",
            "source": metadatas[idx].get("source", "documento desconhecido") if idx < len(metadatas) else "documento desconhecido",
        }
        for idx in range(len(documents))
    ]


# Backward-compat: módulos antigos ainda podem importar EMBED_MODEL.
EMBED_MODEL = LOCAL_EMBED_MODEL


# ── BM25 Retriever ───────────────────────────────────────────────────────────

class BM25Retriever:
    """Índice BM25Okapi sobre o corpus ChromaDB.

    Mantém em memória os textos e fontes de todos os chunks, permitindo
    pesquisa léxica (match exacto de termos, nomes, códigos) que complementa
    a pesquisa vectorial semântica. Combinados via Reciprocal Rank Fusion.

    Persistência: salva/carrega de um ficheiro pickle em backend/db/.
    Actualização incremental: add_documents re-constrói o índice (aceitável
    para corpora de centenas a poucos milhares de chunks).
    """

    def __init__(self):
        self.bm25 = None
        self.docs: List[Dict[str, str]] = []   # [{"text": ..., "source": ...}]

    @property
    def count(self) -> int:
        return len(self.docs)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [w for w in text.split() if w]

    def build_from_collection(self, collection) -> None:
        """Constrói o índice BM25 a partir de todos os documentos na coleção."""
        try:
            result = collection.get(include=["documents", "metadatas"])
            raw_docs = result.get("documents") or []
            raw_metas = result.get("metadatas") or []
        except Exception as exc:
            print(f"[bm25] falhou ao obter documentos: {exc}")
            return

        self.docs = [
            {"text": d or "", "source": (m or {}).get("source", "?")}
            for d, m in zip(raw_docs, raw_metas)
        ]
        self._rebuild_index()
        print(f"[bm25] índice construído com {len(self.docs)} documentos.")

    def add_documents(self, texts: List[str], sources: List[str]) -> None:
        """Adiciona novos documentos e reconstrói o índice."""
        new = [{"text": t or "", "source": s or "?"} for t, s in zip(texts, sources)]
        self.docs.extend(new)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [self._tokenize(d["text"]) for d in self.docs]
            self.bm25 = BM25Okapi(tokenized) if tokenized else None
        except ImportError:
            print("[bm25] rank-bm25 não instalado; pesquisa híbrida desactivada.")
            self.bm25 = None

    def search(self, query: str, k: int = 30) -> List[Tuple[Dict[str, str], float]]:
        """Devolve lista de (chunk_dict, score) ordenada por relevância."""
        if self.bm25 is None or not self.docs:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            (self.docs[i], float(s))
            for i, s in ranked[:k]
            if s > 0 and i < len(self.docs)
        ]

    def save(self, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"bm25": self.bm25, "docs": self.docs}, f)
            print(f"[bm25] índice guardado em {path}")
        except Exception as exc:
            print(f"[bm25] falhou ao guardar: {exc}")

    def load(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data.get("bm25")
            self.docs = data.get("docs") or []
            print(f"[bm25] índice carregado de {path} ({len(self.docs)} docs).")
            return True
        except Exception as exc:
            print(f"[bm25] falhou ao carregar {path}: {exc}")
            return False


def hybrid_query_collection(
    collection,
    question: str,
    k: int = 20,
    bm25_retriever: Optional["BM25Retriever"] = None,
    where: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    """Pesquisa híbrida: BM25 léxico + ChromaDB vectorial → RRF fusion.

    Garante que matches exactos de nomes, códigos e siglas (BM25) e matches
    semânticos (vector) sejam ambos considerados. Se o BM25 não estiver
    disponível, faz pesquisa vectorial pura.

    Args:
        collection: coleção ChromaDB.
        question: pergunta do utilizador.
        k: número final de resultados.
        bm25_retriever: instância de BM25Retriever (None → só vector).
        where: filtro de metadata ChromaDB (opcional).
    """
    n = min(k * 3, max(k + 10, 40))   # candidatos por canal antes de fusão
    rrf_k = 60                          # constante RRF standard

    # ── 1. Pesquisa vectorial ────────────────────────────────────────────
    vector_results: List[Dict[str, str]] = []
    vector_order: Dict[str, int] = {}   # text_key → rank
    result_map: Dict[str, Dict[str, str]] = {}

    try:
        query_kwargs = {"query_texts": [question], "n_results": n}
        if where is not None:
            query_kwargs["where"] = where
        raw = collection.query(**query_kwargs)
        v_docs = (raw.get("documents") or [[]])[0]
        v_metas = (raw.get("metadatas") or [[]])[0]
        for rank, (doc, meta) in enumerate(zip(v_docs, v_metas)):
            if not doc:
                continue
            chunk = {"text": doc, "source": (meta or {}).get("source", "?")}
            key = doc[:120]
            vector_order[key] = rank
            if key not in result_map:
                result_map[key] = chunk
            vector_results.append(chunk)
    except Exception as exc:
        print(f"[hybrid] vector search falhou: {exc}")
        if bm25_retriever is None:
            return []

    # ── 2. Pesquisa BM25 ────────────────────────────────────────────────
    bm25_order: Dict[str, int] = {}

    if bm25_retriever is not None and bm25_retriever.count > 0:
        bm25_hits = bm25_retriever.search(question, k=n)
        for rank, (chunk, _score) in enumerate(bm25_hits):
            if not chunk.get("text"):
                continue
            # Aplicar filtro de metadata (se existir) manualmente para BM25
            if where is not None:
                pass   # BM25 não tem metadata — incluímos sempre e deixamos o LLM filtrar
            key = chunk["text"][:120]
            bm25_order[key] = rank
            if key not in result_map:
                result_map[key] = chunk

    # ── 3. RRF fusion ───────────────────────────────────────────────────
    all_keys = set(vector_order) | set(bm25_order)
    rrf_scores: Dict[str, float] = {}
    for key in all_keys:
        score = 0.0
        if key in vector_order:
            score += 1.0 / (rrf_k + vector_order[key])
        if key in bm25_order:
            score += 1.0 / (rrf_k + bm25_order[key])
        rrf_scores[key] = score

    ranked = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
    return [result_map[key] for key in ranked if key in result_map]
