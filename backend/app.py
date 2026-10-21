import json
import os
import shutil
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional
import chromadb

from backend.answer import compose_answer, contextualize_query
from backend.retrieval import (
    BM25Retriever,
    create_collection,
    query_collection,
    hybrid_query_collection,
    index_documents,
    collection_name_for_provider,
    embed_health,
    list_gemini_models,
)
from backend.ingest import (
    crawl_site,
    extract_markdown_from_pdf,
    extract_text_from_pdf,
    load_pdf_documents,
    scrape_ipb_page,
)
from backend.vision import extract_with_vision, is_scanned_pdf
from backend.chunk import chunk_markdown, chunk_text
from backend.pdf_tables import extract_table_chunks, infer_schema
from backend.schema_store import (
    register_schema,
    remove_schema,
    render_schema_for_prompt,
    aggregate_columns,
)
from backend import conversations as conv_store

DATA_DIR = "backend/data"

HARDCODED_SOURCES: set = set()

SCRAPERS_FILE = os.path.join(DATA_DIR, "scrapers.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
SUPPORTED_EXTS = (".pdf", ".docx", ".html")


def _load_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return {"scraping_enabled": True}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"scraping_enabled": True}


def _save_settings(settings: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def _scraping_enabled() -> bool:
    return bool(_load_settings().get("scraping_enabled", True))


def _load_scrapers() -> list:
    if not os.path.exists(SCRAPERS_FILE):
        return []
    try:
        with open(SCRAPERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_scrapers(entries: list) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCRAPERS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

app = FastAPI(title="EducaAssistAI Backend", version="0.1")

# CORS — permitir o frontend Angular (dev server em :4200) a falar com a API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None  # se presente, persiste a troca na conversa

    @validator("question")
    def question_length(cls, v: str) -> str:
        v = v.strip()
        if len(v) > 2000:
            raise ValueError("Pergunta demasiado longa (máx 2000 caracteres).")
        return v

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: Optional[str] = None
    user_message_id: Optional[str] = None
    assistant_message_id: Optional[str] = None

# Initialize collection. Only (re)index if the collection is empty, para
# evitar re-download do modelo de embeddings e re-scraping do site a cada
# arranque. Para forçar re-indexação, definir REINDEX=1 como env var.

# Nome da coleção depende do provider de embeddings (ipb_docs_v3_local /
# ipb_docs_v3_gemini), pelo que se mantêm dois índices isolados em paralelo.
COLLECTION_NAME = collection_name_for_provider("ipb_docs_v3")
client = chromadb.PersistentClient(path="backend/db")
collection = create_collection(persist_directory="backend/db", name=COLLECTION_NAME)

print(f"Collection '{COLLECTION_NAME}' has {collection.count()} documents.")

# ── BM25 global ─────────────────────────────────────────────────────────────

_BM25_PATH = os.path.join("backend", "db", "bm25_index.pkl")
_bm25: Optional[BM25Retriever] = None


def _get_bm25() -> BM25Retriever:
    """Devolve o retriever BM25 global, construindo-o se necessário."""
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Retriever()
        if os.path.exists(_BM25_PATH):
            ok = _bm25.load(_BM25_PATH)
            if ok and _bm25.count != collection.count():
                print(
                    f"[bm25] índice desactualizado ({_bm25.count} vs "
                    f"{collection.count()} na coleção) — a reconstruir…"
                )
                _bm25.build_from_collection(collection)
                _bm25.save(_BM25_PATH)
        else:
            _bm25.build_from_collection(collection)
            _bm25.save(_BM25_PATH)
    return _bm25


def _rebuild_bm25() -> None:
    """Reconstrói o índice BM25 do zero a partir da coleção actual."""
    global _bm25
    _bm25 = BM25Retriever()
    _bm25.build_from_collection(collection)
    _bm25.save(_BM25_PATH)


def _update_bm25(texts: List[str], sources: List[str]) -> None:
    """Adiciona novos documentos ao índice BM25 e persiste."""
    bm25 = _get_bm25()
    bm25.add_documents(texts, sources)
    bm25.save(_BM25_PATH)


# Pré-aquecimento do LLM em thread separada — carrega o Mistral em memória
# logo no arranque para que a 1ª pergunta do utilizador não pague o custo do
# load do modelo (em CPU, tipicamente 30–90s).
@app.on_event("startup")
def _warmup_models():
    """Pré-carrega LLM (fornecedor configurado em LLM_PROVIDER) e modelo
    de embeddings (sentence-transformers) em threads separadas, para que
    a 1ª query e o 1º upload do utilizador não paguem o custo de
    cold-start. Não bloqueia o arranque do servidor."""
    import threading

    def _warm_llm():
        try:
            from backend.llm_client import get_client

            client = get_client()
            print(
                f"[warmup] A pré-carregar LLM via {client.provider} "
                f"({client.model})..."
            )
            client.warmup()
            print(f"[warmup] LLM ({client.provider}) pronto.")
        except Exception as exc:
            print(f"[warmup] LLM falhou (não-bloqueante): {exc}")

    def _warm_embeddings():
        try:
            print("[warmup] A pré-carregar modelo de embeddings...")
            query_collection(collection, "warmup", k=1)
            print("[warmup] Embeddings prontos.")
        except Exception as exc:
            print(f"[warmup] Embeddings falhou (não-bloqueante): {exc}")

    def _warm_bm25():
        try:
            print("[warmup] A inicializar índice BM25…")
            _get_bm25()
            print("[warmup] BM25 pronto.")
        except Exception as exc:
            print(f"[warmup] BM25 falhou (não-bloqueante): {exc}")

    threading.Thread(target=_warm_llm, daemon=True).start()
    threading.Thread(target=_warm_embeddings, daemon=True).start()
    threading.Thread(target=_warm_bm25, daemon=True).start()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/health/llm")
def health_llm():
    """Estado do fornecedor LLM ativo (Ollama local ou Groq cloud).

    Devolve: provider, configured_model, available_models, e ok=True/False.
    Útil para o frontend mostrar um badge com o LLM em uso e para
    diagnosticar via curl. O fornecedor é controlado pela env var
    ``LLM_PROVIDER``.
    """
    from backend.llm_client import get_client

    return get_client().health()


@app.get("/health/embeddings")
def health_embeddings():
    """Estado do fornecedor de embeddings (sentence-transformers local ou
    Google Gemini cloud) e da coleção ChromaDB associada.

    Útil para o frontend mostrar um badge com o modelo de embeddings em
    uso e a contagem atual de excertos indexados nessa coleção. O
    fornecedor é controlado pela env var ``EMBED_PROVIDER``.
    """
    h = dict(embed_health())
    h["collection"] = COLLECTION_NAME
    try:
        h["chunks_indexed"] = collection.count()
    except Exception as exc:
        h["chunks_indexed"] = 0
        h["count_error"] = str(exc)
    return h


@app.get("/debug/query")
def debug_query(q: str, k: int = 10):
    """Devolve cada etapa do pipeline de resposta para uma pergunta dada,
    sem passar pelo LLM final. Útil para perceber porque é que uma resposta
    é "Não encontrei essa informação...":

      1) Filtros detectados (heurística + tradutor LLM, se aplicável).
      2) Chunks recuperados (texto, fonte, distância semântica).
      3) Distribuição por fonte (que documentos contribuíram).
      4) Prompt completo que SERIA enviado ao LLM (sem o chamar).

    Uso: GET /debug/query?q=quando começam as aulas&k=15
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Pergunta não pode estar vazia.")

    debug: dict = {
        "question": q,
        "collection": COLLECTION_NAME,
        "total_chunks_in_collection": collection.count(),
    }

    # ETAPA 1 — heurística + tradutor LLM (igual ao /query)
    where = None
    filter_origin = "(nenhum)"

    detected = _detect_filters(q)
    if detected is not None:
        where, label = detected
        filter_origin = f"heurística: {label}"

    if where is None:
        cols = aggregate_columns()
        if cols and _question_mentions_schema_values(q, cols):
            from backend.answer import translate_query
            schema_text = render_schema_for_prompt(max_keys=15)
            if schema_text:
                allowed_keys = [c["key"] for c in cols]
                try:
                    llm_filter = translate_query(q, schema_text, allowed_keys)
                    if llm_filter:
                        where = llm_filter
                        filter_origin = "tradutor LLM"
                except Exception as exc:
                    debug["translator_error"] = str(exc)

    debug["filter_origin"] = filter_origin
    debug["where_clause"] = where

    # ETAPA 2 — retrieval
    if where is not None:
        results = _hybrid_retrieve(q, where, limit=k)
        debug["retrieval_mode"] = "híbrido (filtro + ranking semântico)"
        if not results:
            debug["fallback"] = "filtro estrito demais — vector puro como fallback"
            results = query_collection(collection, q, k=k)
    else:
        debug["retrieval_mode"] = "vector puro (sem filtro)"
        results = query_collection(collection, q, k=k)

    # Para diagnóstico, queremos também as distâncias — re-fazemos a query
    # diretamente para obtê-las (sem alterar o caminho normal).
    try:
        raw = collection.query(query_texts=[q], n_results=k, where=where)
        raw_docs = (raw.get("documents") or [[]])[0]
        raw_metas = (raw.get("metadatas") or [[]])[0]
        raw_dists = (raw.get("distances") or [[]])[0]
        debug["chunks"] = [
            {
                "rank": i + 1,
                "distance": float(d) if d is not None else None,
                "source": (m or {}).get("source", "?"),
                "mode": (m or {}).get("mode", "?"),
                "text": (doc or "").strip()[:500],
            }
            for i, (doc, m, d) in enumerate(zip(raw_docs, raw_metas, raw_dists))
        ]
    except Exception as exc:
        debug["chunks_error"] = str(exc)
        debug["chunks"] = [
            {"source": r.get("source", "?"), "text": (r.get("text") or "")[:500]}
            for r in results
        ]

    # Distribuição por fonte para detectar enviesamento
    by_source: dict = {}
    for c in debug.get("chunks", []):
        src = c.get("source", "?")
        by_source[src] = by_source.get(src, 0) + 1
    debug["chunks_by_source"] = by_source

    # ETAPA 3 — prompt que seria enviado ao LLM (sem o chamar, para não gastar quota)
    from backend.answer import _build_context, PROMPT_TEMPLATE, NO_INFO_MSG
    debug["prompt_to_llm"] = PROMPT_TEMPLATE.format(
        context=_build_context(results),
        question=q.strip(),
        no_info=NO_INFO_MSG,
    )
    debug["prompt_chars"] = len(debug["prompt_to_llm"])

    return debug


@app.get("/health/embeddings/models")
def health_embeddings_models():
    """Lista os modelos Gemini a que a ``GOOGLE_API_KEY`` tem acesso.

    Diagnóstico para quando uma chamada à Gemini API falha com
    ``404 model not found`` — devolve quais os modelos disponíveis para
    a chave atual, separando os de embeddings dos de geração. Para o
    utilizador escolher um valor válido para ``GEMINI_EMBED_MODEL`` ou
    ``GROQ_MODEL`` (não, este endpoint só lista Gemini)."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    return list_gemini_models(api_key)

import re as _re

# Padrões para detetar dias mencionados na pergunta. Se o utilizador
# perguntar algo como "quem atende à sexta?" devolvemos TODOS os docentes
# com horário a esse dia, via filtro determinístico no ChromaDB (sem ranking
# por similaridade, sem limite k).
DAY_QUERY_PATTERNS = {
    "has_segunda": [r"\bsegunda(?:s)?(?:-feira)?\b", r"\b2[ªa]?\s*feira\b"],
    "has_terca":   [r"\bter[çc]a(?:s)?(?:-feira)?\b", r"\b3[ªa]?\s*feira\b"],
    "has_quarta":  [r"\bquarta(?:s)?(?:-feira)?\b", r"\b4[ªa]?\s*feira\b"],
    "has_quinta":  [r"\bquinta(?:s)?(?:-feira)?\b", r"\b5[ªa]?\s*feira\b"],
    "has_sexta":   [r"\bsexta(?:s)?(?:-feira)?\b", r"\b6[ªa]?\s*feira\b"],
    "has_sabado":  [r"\bs[áa]bado(?:s)?\b"],
}

DAY_LABEL = {
    "has_segunda": "segunda-feira",
    "has_terca":   "terça-feira",
    "has_quarta":  "quarta-feira",
    "has_quinta":  "quinta-feira",
    "has_sexta":   "sexta-feira",
    "has_sabado":  "sábado",
}

# Departamentos conhecidos no PDF do IPB; ordenados por comprimento
# decrescente para evitar que "DE" matche dentro de "DCES" etc.
KNOWN_DEPS = sorted(
    ["DCES", "DIC", "DTM", "DGI", "DCCP", "DTQB", "DMA", "ESAB", "ESEB", "ESTiG", "DM", "DE"],
    key=len,
    reverse=True,
)


def _extract_day_clauses(q_lower: str):
    """Devolve lista de flags de dias mencionados na pergunta."""
    found = []
    for flag, patterns in DAY_QUERY_PATTERNS.items():
        if any(_re.search(p, q_lower) for p in patterns):
            found.append(flag)
    return found


def _extract_gabinete(question: str):
    """Captura o valor do gabinete/sala mencionado: 'sala 13' → '13'."""
    m = _re.search(
        r"\b(?:sala|gabinete|gab\.?)\s+([A-Za-z0-9&/.\-]+)",
        question,
        _re.IGNORECASE,
    )
    return m.group(1) if m else None


def _extract_dep(question: str):
    """Captura código de departamento mencionado, p.ex. 'do DCES'."""
    for dep in KNOWN_DEPS:
        if _re.search(rf"\b{dep}\b", question):  # case-sensitive
            return dep
    return None


def _detect_filters(question: str):
    """Constrói filtro ChromaDB (where) + label legível, combinando os
    sinais detetáveis na pergunta: dia(s), sala/gabinete, departamento.

    Devolve (where, label) ou None se nada for detetável.
    """
    q_low = question.lower()
    clauses: list = []
    label_parts: list = []

    days = _extract_day_clauses(q_low)
    if len(days) == 1:
        clauses.append({days[0]: True})
        label_parts.append(DAY_LABEL[days[0]])
    elif len(days) > 1:
        clauses.append({"$or": [{f: True} for f in days]})
        label_parts.append(" ou ".join(DAY_LABEL[f] for f in days))

    gab = _extract_gabinete(question)
    if gab:
        clauses.append({"gab": gab})
        label_parts.append(f"gabinete {gab}")

    dep = _extract_dep(question)
    if dep:
        clauses.append({"dep": dep})
        label_parts.append(f"departamento {dep}")

    if not clauses:
        return None

    where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
    label = " e ".join(label_parts)
    return (where, label)


def _format_listing(label: str, docs: list, metas: list) -> tuple:
    """Formata uma listagem determinística como bullet list. Sem LLM."""
    if not docs:
        return (f"Não encontrei resultados para: {label}.", [])

    n = len(docs)
    sources: list = []
    for m in metas:
        s = (m or {}).get("source", "documento desconhecido")
        if s not in sources:
            sources.append(s)

    lines = [
        f"Encontrei {n} resultado{'s' if n != 1 else ''} para: {label}.",
        "",
    ]
    for d in docs:
        text = (d or "").strip()
        if text:
            lines.append(f"- {text}")
    return ("\n".join(lines), sources)




def _question_mentions_schema_values(question: str, cols: list) -> bool:
    """Heurística rápida (sem LLM): a pergunta contém alguma palavra que
    aparece nos sample values do esquema? Se não, faz pouco sentido invocar
    o tradutor LLM porque a pergunta provavelmente não é filtrável.

    IMPORTANTE: usa word boundaries (\\b) para não disparar com falsos
    positivos como "SEG" dentro de "SEGundo" ou "DE" dentro de "DEzembro".
    Amostras com menos de 3 chars são ignoradas — são demasiado ambíguas
    para um match seguro mesmo com fronteiras de palavra.
    """
    q_low = question.lower()
    for col in cols:
        for sample in col.get("samples", []):
            if not isinstance(sample, str):
                continue
            sl = sample.strip().lower()
            if len(sl) < 3:
                continue
            # Match como palavra completa, não como substring.
            if _re.search(rf"\b{_re.escape(sl)}\b", q_low):
                return True
    # Palavras-pivot genéricas ("departamento", "gabinete", "sala"...).
    for hint in ("departamento", "gabinete", "sala", "horário", "horario", "atende", "professor", "docente"):
        if hint in q_low:
            return True
    return False


_FOLLOWUP_CONNECTORS = (
    "e ", "e o ", "e a ", "e os ", "e as ", "e no ", "e na ",
    "e quando ", "e qual ", "e quem ", "e onde ", "e como ",
    "mas ", "então ", "também ", "porquê", "porque ", "e se ",
    "e esse ", "essa ", "esses ", "essas ", "esse ", "aquele ",
)


def _is_followup_question(question: str) -> bool:
    """Heurística: pergunta curta ou inicia com conector — provável follow-up."""
    q = question.strip().lower()
    if any(q.startswith(c) for c in _FOLLOWUP_CONNECTORS):
        return True
    # Pergunta de ≤ 5 palavras sem verbo principal → provavelmente depende do contexto
    words = q.split()
    if len(words) <= 5:
        return True
    return False


def _diverse_rerank(chunks: list, max_per_source: int = 6, final_k: int = 15) -> list:
    """Garante diversidade de fontes sem cortar demasiados chunks relevantes.

    final_k=15: mais hipóteses de incluir o chunk exacto com a resposta.
    max_per_source=6: permite ver mais de um documento sem monopolização.
    Calibrado para 8b-instant (20k TPM): 15×2000 chars ÷ 4 ≈ 7500 tok input.
    """
    from urllib.parse import urlparse as _urlparse
    counts: dict = {}
    result: list = []
    for chunk in chunks:
        src = chunk.get("source", "")
        try:
            p = _urlparse(src)
            group_key = p.netloc if p.scheme in ("http", "https") else src
        except Exception:
            group_key = src
        cnt = counts.get(group_key, 0)
        if cnt < max_per_source:
            counts[group_key] = cnt + 1
            result.append(chunk)
        if len(result) >= final_k:
            break
    return result


# ── Agregação — detecção e retrieval completo ────────────────────────────────

_AGGREGATION_RE = _re.compile(
    r'\bquantos?\b|\btotal\b|\bnúmero\s+de\b'
    r'|\btodos?\s+os\b|\btodas?\s+as\b'
    r'|\blista[r]?\b|\benumera[r]?\b'
    r'|\bcompleto[a]?\b|\binteiro[a]?\b',
    _re.I | _re.UNICODE,
)

# Free tier Groq (14 400 tok/min): chunks curtos (~150 tok) → 80 × 150 = 12 000 tok,
# dentro do limite por minuto. Chunks longos (markdown, ~500 tok) → 80 × 500 = 40 000 tok,
# pode ultrapassar — nesse caso o retrieval semântico dentro da fonte selecciona os mais
# relevantes e adiciona uma nota ao utilizador.
# Sobrescreve com a env var MAX_AGGREGATION_CHUNKS.
MAX_AGGREGATION_CHUNKS = int(os.getenv("MAX_AGGREGATION_CHUNKS", "80"))


def _is_aggregation_query(question: str) -> bool:
    """True se a pergunta exige contagem ou listagem completa de registos."""
    return bool(_AGGREGATION_RE.search(question))


def _full_source_retrieve(source: str) -> list:
    """Devolve TODOS os chunks de uma fonte específica, sem limite top-k.

    Usado para perguntas de agregação (contar, listar todos os itens) onde
    ver apenas os top-k produziria respostas incompletas ou incorrectas.
    """
    try:
        result = collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        return [
            {"text": d, "source": (m or {}).get("source", source)}
            for d, m in zip(docs, metas)
            if d and d.strip()
        ]
    except Exception as exc:
        print(f"[full_source] erro a obter chunks de '{source}': {exc}")
        return []


def _hybrid_retrieve(question: str, where, limit: int = 30) -> list:
    """Retrieval híbrido: aplica filtro de metadata + ranking semântico.

    Faz collection.query (vector + where) para trazer só os chunks que
    passam o filtro, ordenados por relevância à pergunta.
    """
    try:
        res = collection.query(
            query_texts=[question],
            where=where,
            n_results=limit,
        )
    except Exception as exc:
        print(f"[query] hybrid query falhou ({exc}); fallback para .get(where=...)")
        try:
            res_get = collection.get(where=where, limit=limit)
            docs = res_get.get("documents") or []
            metas = res_get.get("metadatas") or []
            return [
                {"text": d, "source": (m or {}).get("source", "documento desconhecido")}
                for d, m in zip(docs, metas)
            ]
        except Exception as exc2:
            print(f"[query] .get(where=...) também falhou ({exc2})")
            return []

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    return [
        {"text": d, "source": (m or {}).get("source", "documento desconhecido")}
        for d, m in zip(docs, metas)
    ]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Pergunta não pode estar vazia.")

    # Se foi passada uma conversation_id, garantimos que existe ANTES de
    # gastar tempo a fazer retrieval/LLM. Se não existe, falha cedo.
    if request.conversation_id is not None:
        conv = conv_store.get_conversation(request.conversation_id, include_messages=False)
        if conv is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversa {request.conversation_id} não encontrada.",
            )

    # Histórico conversacional — as últimas mensagens ANTES da atual.
    # Usado em dois sítios: (1) para contextualizar a query de retrieval
    # quando a pergunta parece ser um follow-up; (2) passado ao LLM para
    # que possa interpretar referências à conversa anterior ("e no 2.º?").
    conv_history: List[Dict] = []
    if request.conversation_id is not None:
        conv_history = conv_store.get_recent_messages(
            request.conversation_id, n=10
        )

    # Persistir a mensagem do utilizador antes do retrieval — se o LLM
    # rebentar, o utilizador ainda vê a sua mensagem na conversa e pode
    # tentar de novo.
    user_message_id: Optional[str] = None
    if request.conversation_id is not None:
        user_msg = conv_store.add_message(
            conversation_id=request.conversation_id,
            role="user",
            content=request.question,
            execution_state="completed",
        )
        if user_msg is not None:
            user_message_id = user_msg.get("id")

    # Query de retrieval — contextualização LLM para follow-ups.
    # Em vez de concatenar ingenuamente com a última pergunta, pede ao LLM
    # que reformule a pergunta actual para ser autónoma (resolve pronomes,
    # referências implícitas, etc.).
    retrieval_question = request.question
    if conv_history and _is_followup_question(request.question):
        contextualized = contextualize_query(request.question, conv_history)
        if contextualized and contextualized.strip() != request.question.strip():
            retrieval_question = contextualized
            print(f"[query] follow-up contextualizado: {retrieval_question[:100]}")

    # Corpus vazio — resposta imediata sem chamar o LLM.
    if collection.count() == 0:
        return QueryResponse(
            answer="Ainda não foi adicionado nenhum documento. Adicione documentos no painel de Administração.",
            sources=[],
            conversation_id=request.conversation_id,
            user_message_id=user_message_id,
        )

    where = None
    filter_origin = ""

    # ETAPA 1 — Heurística rápida (zero LLM): tenta apanhar dia/gab/dep só
    # com regex. Se conseguir, usamos isso (instantâneo, gratuito).
    detected = _detect_filters(request.question)
    if detected is not None:
        where, label = detected
        filter_origin = "heurística"
        print(f"[query] {filter_origin}: {where}  ({label})")

    # ETAPA 2 — Tradutor LLM-driven: só tentamos se a heurística falhou E
    # se a pergunta contém alguma palavra que pareça referir-se aos dados
    # tabulares (algum sample value das colunas registadas). Isto evita
    # gastar 30–60s do LLM em perguntas claramente factuais como "Quem é
    # o orientador?" — que não têm filtro estruturável aplicável.
    if where is None:
        cols = aggregate_columns()
        if cols and _question_mentions_schema_values(request.question, cols):
            from backend.answer import translate_query
            schema_text = render_schema_for_prompt(max_keys=15)
            if schema_text:
                allowed_keys = [c["key"] for c in cols]
                try:
                    llm_filter = translate_query(
                        request.question, schema_text, allowed_keys
                    )
                    if llm_filter:
                        where = llm_filter
                        filter_origin = "tradutor LLM"
                        print(f"[query] {filter_origin}: {where}")
                except Exception as exc:
                    print(f"[query] translate_query falhou: {exc}")
        else:
            print("[query] sem palavras do esquema na pergunta — salta tradutor LLM.")

    # ETAPA 3 — Retrieval híbrido BM25 + vector.
    # Combina pesquisa léxica exacta (BM25) com pesquisa semântica (ChromaDB)
    # via Reciprocal Rank Fusion. Garante matches de nomes, siglas e termos
    # exactos que o vector search puro poderia perder.
    bm25 = _get_bm25()
    if where is not None:
        results = hybrid_query_collection(
            collection, retrieval_question, k=25, bm25_retriever=bm25, where=where
        )
        print(f"[query] híbrido+filtro devolveu {len(results)} chunks.")
        if not results:
            print("[query] filtro sem resultados — fallback para híbrido sem filtro.")
            results = _diverse_rerank(
                hybrid_query_collection(collection, retrieval_question, k=30, bm25_retriever=bm25)
            )
    else:
        raw = hybrid_query_collection(collection, retrieval_question, k=30, bm25_retriever=bm25)
        results = _diverse_rerank(raw)
        print(f"[query] hibrido: {len(raw)} recuperados -> {len(results)} apos diversidade.")

    # ETAPA 3.5 — Upgrade para agregação.
    # Se a pergunta pede contagem ou listagem completa E o retrieval identificou
    # uma fonte relevante, substituir os top-k pela totalidade dos chunks dessa
    # fonte. Isto permite responder correctamente a "quantos professores têm
    # atendimento?" ou "lista todos os docentes do DCES" sem truncar o documento.
    aggregation_note = ""
    if _is_aggregation_query(request.question) and results:
        from collections import Counter as _Counter
        # Preferir fontes locais (PDF/ficheiro) sobre URLs web para agregação.
        pdf_srcs  = [r["source"] for r in results if not r.get("source","").startswith("http")]
        web_srcs  = [r["source"] for r in results if r.get("source","").startswith("http")]
        candidates = pdf_srcs or web_srcs
        if candidates:
            top_source = _Counter(candidates).most_common(1)[0][0]
            full_chunks = _full_source_retrieve(top_source)
            total = len(full_chunks)
            if full_chunks:
                truncated = total > MAX_AGGREGATION_CHUNKS
                if truncated:
                    # Demasiados chunks para o contexto — ordenar por relevância
                    # semântica dentro da fonte e ficar com os mais relevantes.
                    try:
                        ranked = collection.query(
                            query_texts=[retrieval_question],
                            where={"source": top_source},
                            n_results=min(MAX_AGGREGATION_CHUNKS, total),
                        )
                        r_docs  = (ranked.get("documents") or [[]])[0]
                        r_metas = (ranked.get("metadatas") or [[]])[0]
                        full_chunks = [
                            {"text": d, "source": (m or {}).get("source", top_source)}
                            for d, m in zip(r_docs, r_metas) if d and d.strip()
                        ]
                    except Exception:
                        full_chunks = full_chunks[:MAX_AGGREGATION_CHUNKS]
                    aggregation_note = (
                        f"\n\n*(Nota: este documento tem {total} registos; "
                        f"foram analisados {len(full_chunks)} devido ao limite de contexto do modelo. "
                        f"Para maior precisão, filtra por departamento, dia ou outro critério — "
                        f"ou define a variável de ambiente `MAX_AGGREGATION_CHUNKS` para um valor superior.)*"
                    )
                results = full_chunks
                print(
                    f"[query] agregação: {total} chunks de '{top_source}' "
                    f"-> {len(results)} enviados ao LLM"
                    + (" (truncado)" if truncated else " (completo)")
                )

    # ETAPA 4 — Síntese pelo LLM. Passa histórico para que o modelo
    # consiga resolver referências a trocas anteriores da conversa.
    answer, sources = compose_answer(request.question, results, history=conv_history or None)

    # ETAPA 4.5 — Retry dirigido ao documento.
    # Quando o LLM não encontra a resposta mas o retrieval identificou um
    # documento local como relevante, fazemos uma nova query semântica
    # limitada a esse documento (top-15 chunks). Isto resolve o caso em que
    # o chunk exacto com a resposta ficou fora dos top-k iniciais.
    # Custo: 1 chamada LLM extra × ~7500 tok — só ocorre quando necessário.
    if not sources and results:
        top_local_src = next(
            (r.get("source", "") for r in results[:6]
             if r.get("source", "") and not r.get("source", "").startswith("http")),
            None,
        )
        if top_local_src:
            print(f"[query] retry dirigido a '{top_local_src}' (chunk exacto pode estar fora do top-k).")
            try:
                src_raw = collection.query(
                    query_texts=[retrieval_question],
                    where={"source": top_local_src},
                    n_results=15,
                )
                src_docs  = (src_raw.get("documents") or [[]])[0]
                src_metas = (src_raw.get("metadatas") or [[]])[0]
                src_chunks = [
                    {"text": d, "source": (m or {}).get("source", top_local_src)}
                    for d, m in zip(src_docs, src_metas)
                    if d and d.strip()
                ]
                if src_chunks:
                    answer2, sources2 = compose_answer(
                        request.question, src_chunks, history=conv_history or None
                    )
                    if sources2:
                        answer, sources, results = answer2, sources2, src_chunks
                        print(f"[query] retry bem-sucedido: {len(src_chunks)} chunks de '{top_local_src}'")
            except Exception as exc:
                print(f"[query] retry falhou ({exc}); mantendo resposta original.")

    # Acrescentar nota de truncagem se aplicável (só quando houve resposta).
    if aggregation_note and sources:
        answer += aggregation_note

    if not sources:
        # Sugerir as fontes mais próximas para ajudar o utilizador a reformular.
        hint_labels = []
        for r in (results or [])[:4]:
            s = r.get("source", "")
            if not s:
                continue
            label = s.split("/")[-1] if "/" in s else s
            if label not in hint_labels:
                hint_labels.append(label)
        if hint_labels:
            hint = (
                "\n\n*Fontes mais próximas encontradas: "
                + ", ".join(f"`{l}`" for l in hint_labels)
                + ". Tenta reformular mencionando um destes documentos "
                "ou usando um critério mais específico (ex: nome, departamento, dia).*"
            )
        else:
            hint = (
                "\n\n*Dica: tenta reformular a pergunta ou indica o nome "
                "do documento onde sabes que existe essa informação.*"
            )
        answer = "Não encontrei essa informação nos documentos fornecidos." + hint

    # Persistir resposta do assistente com toda a metadata agentic.
    assistant_message_id: Optional[str] = None
    if request.conversation_id is not None:
        # Recolhe info do fornecedor LLM/embed ativo para metadata
        from backend.llm_client import get_client as get_llm_client
        try:
            llm_info = get_llm_client()
            llm_meta = {"provider": llm_info.provider, "model": llm_info.model}
        except Exception:
            llm_meta = {}
        try:
            embed_meta = {
                "provider": embed_health().get("provider"),
                "model": embed_health().get("configured_model"),
            }
        except Exception:
            embed_meta = {}

        tools_used = [{
            "name": "retrieval",
            "mode": "hybrid" if where is not None else "vector",
            "filter_origin": filter_origin or "(nenhum)",
            "where": where,
            "chunks_retrieved": len(results),
        }]
        agent_actions = [
            {"step": "filter_detection", "result": filter_origin or "(nenhum)"},
            {"step": "retrieval", "result": f"{len(results)} chunks"},
            {"step": "synthesis", "result": "ok" if sources else "no_info"},
        ]

        assistant_msg = conv_store.add_message(
            conversation_id=request.conversation_id,
            role="assistant",
            content=answer,
            sources=sources,
            tools_used=tools_used,
            agent_actions=agent_actions,
            execution_state="completed" if sources else "no_info",
            metadata={"llm": llm_meta, "embed": embed_meta},
        )
        if assistant_msg is not None:
            assistant_message_id = assistant_msg.get("id")

    return QueryResponse(
        answer=answer,
        sources=sources,
        conversation_id=request.conversation_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )


# ── /retrieve — retrieval-only endpoint (para avaliação e debugging) ─────────

class RetrieveRequest(BaseModel):
    question: str
    k: int = 10

    @validator("question")
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Pergunta não pode estar vazia.")
        return v[:2000]

    @validator("k")
    def k_in_range(cls, v: int) -> int:
        return max(1, min(v, 50))


class RetrieveChunk(BaseModel):
    text: str
    source: str


class RetrieveResponse(BaseModel):
    chunks: List[RetrieveChunk]
    n_retrieved: int
    filter_used: str


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(request: RetrieveRequest):
    """Devolve os top-k chunks mais relevantes sem invocar o LLM.

    Replica a lógica de retrieval de /query (filtro heurístico + híbrido ou
    vector puro) mas para antes da síntese. Útil para avaliação standalone
    da qualidade de retrieval e para debugging.
    """
    question = request.question

    # Heurística de filtro (mesma lógica do endpoint /query)
    where = None
    filter_used = "vector"
    detected = _detect_filters(question)
    if detected is not None:
        where, label = detected
        filter_used = f"heurística ({label})"

    # Retrieval híbrido BM25 + vector
    bm25 = _get_bm25()
    if where is not None:
        chunks = hybrid_query_collection(collection, question, k=request.k, bm25_retriever=bm25, where=where)
        filter_used = f"híbrido ({label if detected else 'filtro'})"
        if not chunks:
            chunks = hybrid_query_collection(collection, question, k=request.k, bm25_retriever=bm25)
            filter_used += " → fallback híbrido sem filtro"
    else:
        chunks = hybrid_query_collection(collection, question, k=request.k, bm25_retriever=bm25)

    return RetrieveResponse(
        chunks=[RetrieveChunk(text=c.get("text", ""), source=c.get("source", "")) for c in chunks],
        n_retrieved=len(chunks),
        filter_used=filter_used,
    )


def _extract_text_from_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_text_from_html(path: str) -> str:
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    elements = soup.select("h1, h2, h3, p, li")
    return "\n".join(el.get_text(separator=" ", strip=True) for el in elements)


def _table_extraction_is_useful(table_chunks: List, threshold: float = 0.5) -> bool:
    """Heurística para decidir se a extração tabular produziu chunks
    semanticamente úteis ou se é melhor cair para markdown.

    Critério: a maioria dos chunks tem pelo menos 2 palavras "narrativas"
    (≥ 4 caracteres alfabéticos, sem dígitos). Chunks como "Prof. Silva
    (DCES, sexta-feira, 14:30)" passam; "SEG (1, 1, F, 2, ER, 2, 1)" falha
    porque só tem códigos curtos.

    Retorna True se >= ``threshold`` dos chunks são úteis.
    """
    if not table_chunks:
        return False

    useful = 0
    for entry in table_chunks:
        text = entry[0] if isinstance(entry, tuple) else entry
        words = (text or "").split()
        long_alpha = sum(
            1 for w in words
            if len(w.strip(".,:;()")) >= 4
            and w.replace("-", "").replace("í", "i").replace("ã", "a").isalpha()
        )
        if long_alpha >= 2:
            useful += 1

    quality = useful / len(table_chunks)
    print(f"[upload] Qualidade tabular: {useful}/{len(table_chunks)} = {quality:.2f}")
    return quality >= threshold


class ColumnInfo(BaseModel):
    """Descrição de uma coluna/chave detetada num PDF tabular."""
    key: str              # ex: "col_dep", "has_sexta", "gab"
    type: str             # "text" ou "boolean"
    samples: List = []    # até 3 valores-exemplo


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks: int
    extraction_mode: str = "text"   # "markdown" | "table" | "hybrid" | "vision" | "text"
    row_count: int = 0
    columns: List[ColumnInfo] = []  # vazio se extração foi por texto


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Recebe um PDF, grava-o em backend/data/, extrai-lhe o texto, divide
    em chunks e adiciona-os à coleção sem destruir o que já lá está."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Ficheiro sem nome.")

    fname_lower = file.filename.lower()
    if not any(fname_lower.endswith(ext) for ext in SUPPORTED_EXTS):
        raise HTTPException(
            status_code=400,
            detail="Apenas ficheiros PDF, DOCX e HTML são aceites.",
        )

    safe_name = os.path.basename(file.filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    dest_path = os.path.join(DATA_DIR, safe_name)

    # Grava o ficheiro em disco. Em Windows, se o ficheiro destino já
    # existir e estiver aberto noutro processo (Adobe Reader, Edge,
    # antivírus a fazer scan...), open(..., "wb") falha com PermissionError.
    # Tratamento: tentar uma vez, e em caso de bloqueio fallback para um
    # nome único com prefixo uuid (assim o upload nunca falha por colisão).
    try:
        try:
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(file.file, out)
        except PermissionError:
            print(
                f"[upload] '{dest_path}' está bloqueado/em uso. "
                f"A gravar com nome alternativo."
            )
            # rebobinar o stream do upload, senão fica vazio na 2ª escrita
            try:
                file.file.seek(0)
            except Exception:
                pass
            alt_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
            dest_path = os.path.join(DATA_DIR, alt_name)
            with open(dest_path, "wb") as out:
                shutil.copyfileobj(file.file, out)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Não foi possível gravar o ficheiro em disco: {exc}. "
                    "Verifique se o PDF não está aberto noutro programa "
                    "(Adobe Reader, browser, etc.)."
                ),
            )
    finally:
        await file.close()

    chunks: List[str] = []
    extraction_mode = "text"
    row_metadatas: List[dict] = []
    chunk_modes: List[str] = []   # modo por chunk; vazio → usa extraction_mode global

    if safe_name.lower().endswith(".docx"):
        try:
            text = _extract_text_from_docx(dest_path)
            chunks = chunk_text(text)
            row_metadatas = [{} for _ in chunks]
            extraction_mode = "text"
            print(f"[upload] {len(chunks)} chunks de DOCX.")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Falha a extrair DOCX: {exc}")
    elif safe_name.lower().endswith(".html"):
        try:
            text = _extract_text_from_html(dest_path)
            chunks = chunk_text(text)
            row_metadatas = [{} for _ in chunks]
            extraction_mode = "text"
            print(f"[upload] {len(chunks)} chunks de HTML.")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Falha a extrair HTML: {exc}")
    else:
        # PDF: pipeline ADITIVO — extrai texto completo (markdown) SEMPRE,
        # e adiciona chunks tabulares com metadata estruturada QUANDO disponíveis.
        # Nunca descarta texto em favor das tabelas.
        table_chunks = None

        # ── 1) Extração tabular (opcional) ───────────────────────────────────
        # Produz 1 chunk por linha de tabela, com metadata col_*/has_* para
        # filtros determinísticos (ex: "horário do Prof. X à sexta-feira").
        try:
            table_chunks = extract_table_chunks(dest_path)
        except Exception as exc:
            print(f"[upload] Extração tabular falhou ({exc}).")
            table_chunks = None

        tables_useful = bool(table_chunks and _table_extraction_is_useful(table_chunks))

        if tables_useful:
            try:
                schema = infer_schema(table_chunks)
                register_schema(safe_name, schema)
                print(
                    f"[upload] Esquema registado: {len(schema['columns'])} chaves "
                    f"({safe_name})."
                )
            except Exception as exc:
                print(f"[upload] Falha a registar esquema: {exc}")
        else:
            try:
                remove_schema(safe_name)
            except Exception:
                pass
            if table_chunks:
                print(
                    f"[upload] Extração tabular: {len(table_chunks)} chunks de "
                    f"baixa qualidade — não adicionado."
                )

        # ── 2) Extração de texto completo — SEMPRE obrigatória ───────────────
        # Captura todo o conteúdo: parágrafos, cabeçalhos, listas e tabelas
        # como markdown. É a base de todos os documentos.
        base_mode = "markdown"
        base_chunks: List[str] = []
        base_metadatas: List[dict] = []

        try:
            md = extract_markdown_from_pdf(dest_path)
            base_chunks = chunk_markdown(md)
            base_metadatas = [{} for _ in base_chunks]
            print(f"[upload] {len(base_chunks)} chunks markdown (texto completo).")
        except Exception as exc:
            print(f"[upload] Markdown falhou ({exc}).")
            vision_attempted = False
            if is_scanned_pdf(dest_path) and os.getenv("GOOGLE_API_KEY", "").strip():
                print("[upload] PDF digitalizado detetado — a tentar extração visual.")
                try:
                    md = extract_with_vision(dest_path)
                    base_chunks = chunk_markdown(md)
                    base_metadatas = [{} for _ in base_chunks]
                    base_mode = "vision"
                    vision_attempted = True
                    print(f"[upload] {len(base_chunks)} chunks via Gemini Vision.")
                except Exception as ve:
                    print(f"[upload] Vision falhou ({ve}); fallback para texto.")
            if not vision_attempted:
                try:
                    text = extract_text_from_pdf(dest_path)
                except Exception as exc2:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Falha a extrair PDF: {exc2}",
                    )
                base_chunks = chunk_text(text)
                base_metadatas = [{} for _ in base_chunks]
                base_mode = "text"
                print(f"[upload] {len(base_chunks)} chunks como texto plano (fallback).")

        # ── 3) Combinar: chunks de texto + chunks tabulares (aditivos) ────────
        if tables_useful:
            t_texts = [t for t, _ in table_chunks]
            t_metas = [m for _, m in table_chunks]
            chunks = base_chunks + t_texts
            row_metadatas = base_metadatas + t_metas
            chunk_modes = [base_mode] * len(base_chunks) + ["table"] * len(t_texts)
            extraction_mode = "hybrid" if base_chunks else "table"
            print(
                f"[upload] Pipeline aditivo: {len(base_chunks)} chunks {base_mode} "
                f"+ {len(t_texts)} linhas tabulares = {len(chunks)} total."
            )
        else:
            chunks = base_chunks
            row_metadatas = base_metadatas
            chunk_modes = [base_mode] * len(base_chunks)
            extraction_mode = base_mode

    if not chunks:
        return UploadResponse(
            filename=safe_name,
            chunks_added=0,
            total_chunks=collection.count(),
        )

    # Se este ficheiro já tinha sido carregado antes, apaga os chunks
    # antigos para que um re-upload substitua em vez de acumular.
    try:
        collection.delete(where={"source": safe_name})
    except Exception as exc:
        print(f"[upload] Aviso: não consegui limpar chunks antigos de {safe_name}: {exc}")

    # Adiciona ao índice de forma incremental. Usar uuid no id evita colisões.
    upload_id = uuid.uuid4().hex[:8]
    ids = [f"upload-{upload_id}-{i}" for i in range(len(chunks))]
    # chunk_modes pode ter modos distintos por chunk (ex: hybrid = markdown + table).
    # Se vazio (DOCX/HTML), usa o extraction_mode global para todos.
    effective_modes = chunk_modes if chunk_modes else [extraction_mode] * len(chunks)
    metadatas = [
        {"source": safe_name, "mode": effective_modes[i], **(row_metadatas[i] or {})}
        for i in range(len(chunks))
    ]

    try:
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha a indexar: {exc}")

    # Actualizar índice BM25 com os novos chunks (em background para não
    # bloquear a resposta ao utilizador).
    import threading as _threading
    _threading.Thread(
        target=_update_bm25,
        args=(chunks, [safe_name] * len(chunks)),
        daemon=True,
    ).start()

    # Construir descrição do esquema detetado para devolver ao frontend.
    # Para texto plano não há colunas — devolve lista vazia.
    columns_info: List[ColumnInfo] = []
    row_count = 0
    if extraction_mode in ("table", "hybrid"):
        try:
            from backend.schema_store import load_all_schemas
            schemas = load_all_schemas()
            sch = schemas.get(safe_name, {})
            row_count = sch.get("row_count", len(chunks))
            for col in sch.get("columns", []):
                columns_info.append(
                    ColumnInfo(
                        key=col.get("key", ""),
                        type=col.get("type", "text"),
                        samples=col.get("samples", [])[:3],
                    )
                )
        except Exception as exc:
            print(f"[upload] Falha a serializar esquema na resposta: {exc}")

    return UploadResponse(
        filename=safe_name,
        chunks_added=len(chunks),
        total_chunks=collection.count(),
        extraction_mode=extraction_mode,
        row_count=row_count,
        columns=columns_info,
    )


@app.get("/schemas")
def list_schemas():
    """Devolve todos os esquemas registados (um por PDF tabular).

    Útil para o frontend mostrar "que documentos tenho indexados e quais
    são as chaves pesquisáveis em cada um".
    """
    from backend.schema_store import load_all_schemas
    return load_all_schemas()


class DocumentInfo(BaseModel):
    """Resumo de um documento carregado pelo utilizador."""
    filename: str
    size: int = 0              # bytes
    chunks: int = 0            # nº de excertos indexados na coleção
    mode: str = "text"         # modo de extração: text | markdown | table | vision
    modified_at: str = ""      # ISO 8601 UTC do mtime do ficheiro ("" se desconhecido)


def _collection_chunks_for(source: str) -> tuple:
    """Devolve (n_chunks, mode) para um dado source na coleção.
    Prioridade: vision > hybrid (markdown+table) > table > markdown > text."""
    try:
        res = collection.get(where={"source": source})
    except Exception as exc:
        print(f"[documents] erro a consultar chunks de {source}: {exc}")
        return (0, "text")

    ids = res.get("ids") or []
    metas = res.get("metadatas") or []
    modes_found: set = {(m or {}).get("mode", "text") for m in metas}

    if "vision" in modes_found:
        return (len(ids), "vision")
    # hybrid = documento tem tanto chunks markdown/text como chunks table
    text_modes = modes_found & {"markdown", "text"}
    if "table" in modes_found and text_modes:
        return (len(ids), "hybrid")
    if "table" in modes_found:
        return (len(ids), "table")
    if "markdown" in modes_found:
        return (len(ids), "markdown")
    return (len(ids), "text")


def _safe_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _safe_mtime(path: str) -> str:
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(timespec="seconds")
    except OSError:
        return ""


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    """Lista todos os documentos em ``backend/data/`` com o número de
    excertos indexados e o modo de extração de cada um.
    """
    if not os.path.isdir(DATA_DIR):
        return []

    docs: List[DocumentInfo] = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not any(fname.lower().endswith(ext) for ext in SUPPORTED_EXTS):
            continue
        path = os.path.join(DATA_DIR, fname)
        n_chunks, mode = _collection_chunks_for(fname)
        docs.append(DocumentInfo(
            filename=fname,
            size=_safe_size(path),
            chunks=n_chunks,
            mode=mode,
            modified_at=_safe_mtime(path),
        ))
    return docs


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Apaga um documento: remove o ficheiro do disco, os chunks no
    ChromaDB e o esquema registado (se existir).
    """
    safe_name = os.path.basename(filename)
    if safe_name in HARDCODED_SOURCES:
        raise HTTPException(
            status_code=400,
            detail="Este documento é fixo do sistema e não pode ser apagado.",
        )

    path = os.path.join(DATA_DIR, safe_name)

    # 1) Chunks na coleção
    try:
        collection.delete(where={"source": safe_name})
    except Exception as exc:
        print(f"[delete] aviso: falha a apagar chunks de {safe_name}: {exc}")

    # 2) Esquema (se PDF tabular)
    try:
        remove_schema(safe_name)
    except Exception as exc:
        print(f"[delete] aviso: falha a remover esquema de {safe_name}: {exc}")

    # 3) Ficheiro em disco
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Não foi possível remover o ficheiro do disco: {exc}. "
                    "Verifique se o PDF não está aberto noutro programa."
                ),
            )

    return {
        "deleted": safe_name,
        "total_chunks": collection.count(),
    }


class ReextractRequest(BaseModel):
    """Pedido para re-indexar um PDF já carregado, forçando o modo de extração."""
    mode: str = "hybrid"  # "hybrid" | "markdown" | "text" | "table" | "vision"


@app.post("/documents/{filename}/reextract")
def reextract_document(filename: str, body: ReextractRequest):
    """Re-extrai e re-indexa um PDF que já está em ``backend/data/``,
    forçando o modo de extração. Útil quando a deteção automática produz
    chunks de fraca qualidade — por exemplo, calendários extraídos como
    tabela perdem os cabeçalhos das colunas e ficam ilegíveis
    (``SEG (15, INÍCIO, 13, 10, ...)``). Re-extrair como ``markdown``
    recupera a estrutura completa com cabeçalhos, prosa e tabelas em
    formato unificado.

    Modos suportados:
      * ``hybrid`` (recomendado para PDFs mistos) — combina markdown e
        extração tabular: indexa todo o texto E adiciona chunks por linha
        de tabela com metadata filtrável.
      * ``markdown`` — PyMuPDF4LLM converte para Markdown unificado.
        Ideal para PDFs só de texto ou quando hybrid não for necessário.
      * ``text`` — texto plano, sem estrutura. Fallback se markdown falhar.
      * ``table`` — extração tabular linha-a-linha exclusiva. Útil para
        PDFs que são apenas horários/listas tabulares.
      * ``vision`` — Gemini processa cada página como imagem. Ideal para
        PDFs digitalizados. Requer GOOGLE_API_KEY.

    Apaga os chunks antigos da coleção e o esquema (se existia), e adiciona
    os chunks novos com a mesma estrutura de IDs do upload normal.
    """
    safe_name = os.path.basename(filename)
    if safe_name in HARDCODED_SOURCES:
        raise HTTPException(
            status_code=400,
            detail="Este documento é fixo do sistema e não pode ser re-extraído.",
        )

    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="A re-extração está disponível apenas para ficheiros PDF.",
        )

    mode = (body.mode or "").strip().lower()
    if mode not in ("markdown", "text", "table", "vision", "hybrid"):
        raise HTTPException(
            status_code=400,
            detail="Modo inválido. Use 'hybrid', 'markdown', 'text', 'table' ou 'vision'.",
        )

    path = os.path.join(DATA_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    # 1) Apagar estado antigo: chunks na coleção + esquema registado.
    try:
        collection.delete(where={"source": safe_name})
    except Exception as exc:
        print(f"[reextract] aviso: falha a apagar chunks antigos: {exc}")
    try:
        remove_schema(safe_name)
    except Exception as exc:
        print(f"[reextract] aviso: falha a remover esquema antigo: {exc}")

    # 2) Re-extrair conforme o modo pedido.
    chunks: List[str] = []
    row_metadatas: List[dict] = []
    reextract_chunk_modes: List[str] = []   # modo por chunk (para hybrid)

    if mode in ("table", "hybrid"):
        try:
            table_chunks = extract_table_chunks(path)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Falha a extrair tabelas: {exc}",
            )
        if not table_chunks and mode == "table":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Nenhuma tabela detectada neste PDF. "
                    "Tenta de novo com mode='hybrid', 'markdown' ou 'text'."
                ),
            )
        if table_chunks:
            try:
                schema = infer_schema(table_chunks)
                register_schema(safe_name, schema)
            except Exception as exc:
                print(f"[reextract] falha a registar esquema: {exc}")

        if mode == "table":
            chunks = [t for t, _ in table_chunks]
            row_metadatas = [m for _, m in table_chunks]
            reextract_chunk_modes = ["table"] * len(chunks)
        else:
            # hybrid: markdown como base + linhas tabulares adicionais
            try:
                md = extract_markdown_from_pdf(path)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Falha a extrair markdown para hybrid: {exc}.",
                )
            md_chunks = chunk_markdown(md)
            t_texts = [t for t, _ in table_chunks] if table_chunks else []
            t_metas = [m for _, m in table_chunks] if table_chunks else []
            chunks = md_chunks + t_texts
            row_metadatas = [{} for _ in md_chunks] + t_metas
            reextract_chunk_modes = ["markdown"] * len(md_chunks) + ["table"] * len(t_texts)
            print(
                f"[reextract] hybrid: {len(md_chunks)} chunks markdown "
                f"+ {len(t_texts)} linhas tabulares = {len(chunks)} total."
            )
    elif mode == "vision":
        key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "GOOGLE_API_KEY não definida. "
                    "Necessária para extração visual com Gemini."
                ),
            )
        try:
            md = extract_with_vision(path, api_key=key)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Falha na extração visual: {exc}",
            )
        chunks = chunk_markdown(md)
        row_metadatas = [{} for _ in chunks]
        reextract_chunk_modes = ["vision"] * len(chunks)
    elif mode == "markdown":
        try:
            md = extract_markdown_from_pdf(path)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Falha a extrair markdown: {exc}. "
                    "Tenta de novo com mode='text'."
                ),
            )
        chunks = chunk_markdown(md)
        row_metadatas = [{} for _ in chunks]
        reextract_chunk_modes = ["markdown"] * len(chunks)
    else:  # mode == "text"
        try:
            text = extract_text_from_pdf(path)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Falha a extrair texto: {exc}",
            )
        chunks = chunk_text(text)
        row_metadatas = [{} for _ in chunks]
        reextract_chunk_modes = ["text"] * len(chunks)

    if not chunks:
        return {
            "reextracted": safe_name,
            "mode": mode,
            "chunks_added": 0,
            "total_chunks": collection.count(),
        }

    # 3) Re-indexar com IDs novos para evitar colisões.
    upload_id = uuid.uuid4().hex[:8]
    ids = [f"reextract-{upload_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": safe_name, "mode": reextract_chunk_modes[i], **(row_metadatas[i] or {})}
        for i in range(len(chunks))
    ]
    try:
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha a indexar: {exc}")

    return {
        "reextracted": safe_name,
        "mode": mode,
        "chunks_added": len(chunks),
        "total_chunks": collection.count(),
    }


@app.get("/documents/{filename}/file")
def serve_document(filename: str):
    """Serve o PDF para visualização inline (iframe) ou nova aba do browser."""
    safe_name = os.path.basename(filename)
    path = os.path.join(DATA_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    ext = safe_name.rsplit(".", 1)[-1].lower()
    media_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "html": "text/html",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


class RenameRequest(BaseModel):
    """Pedido para renomear um documento já carregado."""
    new_filename: str


def _sanitize_pdf_name(raw: str) -> str:
    """Garante que o novo nome é um PDF seguro (sem path traversal)."""
    name = os.path.basename(raw or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Nome em branco.")
    # Remove caracteres problemáticos em sistemas de ficheiros (Windows + Linux).
    forbidden = '<>:"/\\|?*\0'
    for ch in forbidden:
        name = name.replace(ch, "_")
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    if name.startswith("."):
        raise HTTPException(status_code=400, detail="Nome inválido.")
    return name


@app.patch("/documents/{filename}/rename")
def rename_document(filename: str, body: RenameRequest):
    """Renomeia um documento: ficheiro em disco + metadata 'source' nos
    chunks indexados + chave no registo de esquemas (se PDF tabular).

    Em caso de falha a meio, tenta-se reverter para um estado consistente.
    """
    safe_old = os.path.basename(filename)
    if safe_old in HARDCODED_SOURCES:
        raise HTTPException(
            status_code=400,
            detail="Este documento é fixo do sistema e não pode ser renomeado.",
        )

    safe_new = _sanitize_pdf_name(body.new_filename)
    if safe_new in HARDCODED_SOURCES:
        raise HTTPException(
            status_code=400,
            detail="Esse nome está reservado pelo sistema.",
        )

    old_path = os.path.join(DATA_DIR, safe_old)
    new_path = os.path.join(DATA_DIR, safe_new)

    if not os.path.exists(old_path):
        raise HTTPException(status_code=404, detail="Documento não encontrado.")

    if safe_old == safe_new:
        # Sem alterações: idempotente, devolver sem trabalho.
        return {"old": safe_old, "renamed": safe_new}

    if os.path.exists(new_path):
        raise HTTPException(
            status_code=409,
            detail=f"Já existe um documento chamado '{safe_new}'.",
        )

    # 1) Renomear ficheiro em disco.
    try:
        os.rename(old_path, new_path)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Não foi possível renomear o ficheiro: {exc}. "
                "Verifique se o PDF não está aberto noutro programa."
            ),
        )

    # 2) Atualizar metadata 'source' em todos os chunks na coleção.
    try:
        res = collection.get(where={"source": safe_old})
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        if ids:
            new_metas = []
            for m in metas:
                m2 = dict(m or {})
                m2["source"] = safe_new
                new_metas.append(m2)
            collection.update(ids=ids, metadatas=new_metas)
    except Exception as exc:
        # Rollback do rename em disco para manter consistência.
        try:
            os.rename(new_path, old_path)
        except OSError as rb_exc:
            print(f"[rename] rollback falhou: {rb_exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Falha a atualizar índice: {exc}",
        )

    # 3) Migrar entrada no schema_store (se este PDF era tabular).
    try:
        from backend.schema_store import load_all_schemas
        schemas = load_all_schemas()
        if safe_old in schemas:
            schema = schemas[safe_old]
            register_schema(safe_new, schema)
            remove_schema(safe_old)
    except Exception as exc:
        print(f"[rename] aviso: falha a migrar esquema: {exc}")

    return {"old": safe_old, "renamed": safe_new}


# ============================================================================
# Conversas & mensagens — chat persistente (Fase 1 do feature de histórico)
# ============================================================================
#
# Camada de persistência em ``backend/conversations.py`` (SQLite). Esta secção
# expõe os endpoints REST que o frontend vai consumir.
#
# Convenção de payloads:
#   - Conversas têm campos serializáveis simples (id, title, timestamps, flags
#     pinned/archived, message_count).
#   - Mensagens têm campos agentic ricos (sources, tools_used, tokens,
#     agent_actions, execution_state, metadata) — todos opcionais no input.
# ============================================================================


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None


class UpdateConversationRequest(BaseModel):
    """Atualização parcial — só os campos enviados são tocados."""
    title: Optional[str] = None
    pinned: Optional[bool] = None
    archived: Optional[bool] = None


class AddMessageRequest(BaseModel):
    role: str = Field(..., description="'user', 'assistant' ou 'system'")
    content: str
    sources: Optional[List[str]] = None
    tools_used: Optional[List[Dict[str, Any]]] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    agent_actions: Optional[List[Dict[str, Any]]] = None
    execution_state: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@app.post("/conversations")
def create_conversation_endpoint(body: CreateConversationRequest):
    """Cria uma conversa nova vazia. ``title`` é opcional — fica "Nova conversa"
    por defeito; o frontend pode chamar mais tarde o endpoint de auto-rename
    para o LLM gerar um título baseado no conteúdo (Fase 2)."""
    conv = conv_store.create_conversation(title=body.title)
    return conv


@app.get("/conversations")
def list_conversations_endpoint(
    include_archived: bool = False,
    grouped: bool = False,
):
    """Lista todas as conversas. Ordenadas por ``pinned DESC, updated_at DESC``.

    Se ``grouped=true``, devolve já agrupado em buckets temporais
    (``pinned``, ``today``, ``yesterday``, ``last_7_days``, ``last_30_days``,
    ``older``) — pronto para o sidebar do frontend consumir sem processamento.
    """
    items = conv_store.list_conversations(include_archived=include_archived)
    if grouped:
        return conv_store.group_conversations_by_time(items)
    return items


@app.get("/conversations/search")
def search_conversations_endpoint(q: str, limit: int = 50):
    """Pesquisa conversas pelo título ou pelo conteúdo das mensagens.

    Cada resultado inclui um campo ``match`` com onde (``title`` ou
    ``message``) ocorreu o primeiro match e um snippet à volta da palavra
    encontrada, pronto para o frontend mostrar.

    IMPORTANTE: este endpoint TEM de estar registado ANTES do
    ``/conversations/{conversation_id}`` para o FastAPI não interpretar
    "search" como um ID — o FastAPI matcha pela ordem de registo.
    """
    return conv_store.search_conversations(q, limit=limit)


@app.get("/conversations/{conversation_id}")
def get_conversation_endpoint(conversation_id: str, include_messages: bool = True):
    """Obtém uma conversa pelo ID, com ou sem o array completo de mensagens."""
    conv = conv_store.get_conversation(conversation_id, include_messages=include_messages)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")
    return conv


@app.patch("/conversations/{conversation_id}")
def update_conversation_endpoint(conversation_id: str, body: UpdateConversationRequest):
    """Atualização parcial: rename, pin/unpin, archive/unarchive."""
    conv = conv_store.update_conversation(
        conversation_id,
        title=body.title,
        pinned=body.pinned,
        archived=body.archived,
    )
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")
    return conv


@app.delete("/conversations/{conversation_id}")
def delete_conversation_endpoint(conversation_id: str):
    """Apaga a conversa e todas as suas mensagens (cascade)."""
    if not conv_store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")
    return {"deleted": conversation_id}


@app.post("/conversations/{conversation_id}/messages")
def add_message_endpoint(conversation_id: str, body: AddMessageRequest):
    """Adiciona uma mensagem a uma conversa existente. Atualiza ``updated_at``
    da conversa, o que afeta a sua posição na lista agrupada por tempo."""
    try:
        msg = conv_store.add_message(
            conversation_id=conversation_id,
            role=body.role,
            content=body.content,
            sources=body.sources,
            tools_used=body.tools_used,
            tokens_input=body.tokens_input,
            tokens_output=body.tokens_output,
            agent_actions=body.agent_actions,
            execution_state=body.execution_state,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if msg is None:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")
    return msg


@app.get("/conversations/{conversation_id}/messages")
def list_messages_endpoint(conversation_id: str):
    """Lista mensagens de uma conversa em ordem cronológica.

    Ligeiramente mais leve que GET /conversations/{id} porque não inclui
    o objeto da conversa em si — útil para paginação futura ou para o
    frontend re-fetch das mensagens após uma resposta sem pedir tudo de novo.
    """
    if conv_store.get_conversation(conversation_id, include_messages=False) is None:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")
    return conv_store.list_messages(conversation_id)


# ----------------------------------------------------------------------
# Fase 2 — auto-rename via LLM
# (a rota /conversations/search está registada acima, antes da dinâmica
#  /conversations/{id}, para evitar colisão de routing)
# ----------------------------------------------------------------------


# Prompt curto para o LLM gerar um título da conversa. Mantido fora da
# função para ser fácil de ajustar (a qualidade do título depende muito
# desta string e dos exemplos).
_TITLE_PROMPT = """És um sistema que gera títulos curtos e descritivos para \
conversas de utilizadores com um assistente documental.

REGRAS:
1. Devolve APENAS o título, sem aspas, sem prefixos como "Título:" ou \
"Conversa sobre".
2. Máximo 6 palavras.
3. Em português europeu.
4. Captura o tópico principal da pergunta.

EXEMPLOS:
Pergunta: "Quando começam as aulas do segundo semestre?"
Título: Início das aulas 2.º semestre

Pergunta: "Quem é o orientador do projeto EducaAssistAI?"
Título: Orientador do projeto

Pergunta: "Qual o horário de atendimento do professor Pedro Lopes na sexta?"
Título: Atendimento sexta — Pedro Lopes

Pergunta: {question}

Título:"""


def _clean_generated_title(raw: str) -> str:
    """Normaliza o output do LLM: remove aspas, prefixos comuns, newlines."""
    title = (raw or "").strip()
    # Remover marcas em PT/EN que o LLM por vezes inclui mesmo proibido
    for prefix in ("Título:", "Titulo:", "Title:", "Conversa sobre:", "Tópico:"):
        if title.lower().startswith(prefix.lower()):
            title = title[len(prefix):].strip()
    title = title.strip(' "\'\n\r\t')
    # Cortar pelo 1º newline (modelos por vezes geram explicação a seguir)
    if "\n" in title:
        title = title.split("\n", 1)[0].strip()
    # Limite conservador
    if len(title) > 80:
        title = title[:77].rstrip() + "..."
    return title


@app.post("/conversations/{conversation_id}/auto-title")
def auto_title_conversation(conversation_id: str):
    """Pede ao LLM um título conciso para esta conversa baseado na primeira
    pergunta do utilizador, e atualiza o campo ``title``.

    O frontend deve chamar este endpoint depois da primeira resposta do
    assistente (não antes — assim o LLM tem contexto suficiente). É um
    endpoint idempotente; cada chamada re-pergunta ao LLM.
    """
    conv = conv_store.get_conversation(conversation_id, include_messages=True)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversa não encontrada.")

    user_msgs = [m for m in (conv.get("messages") or []) if m.get("role") == "user"]
    if not user_msgs:
        raise HTTPException(
            status_code=400,
            detail="Conversa ainda sem mensagens do utilizador — nada para resumir.",
        )

    first_question = user_msgs[0]["content"]

    from backend.llm_client import get_client, LLMTimeout, LLMUnavailable, LLMError

    prompt = _TITLE_PROMPT.format(question=first_question)
    try:
        raw = get_client().generate(
            prompt,
            temperature=0.3,
            num_predict=32,
            timeout=30,
        )
    except (LLMTimeout, LLMUnavailable, LLMError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM indisponível para gerar título: {exc}",
        )

    title = _clean_generated_title(raw)
    if not title:
        raise HTTPException(
            status_code=500,
            detail="LLM devolveu título vazio. Tenta de novo.",
        )

    updated = conv_store.update_conversation(conversation_id, title=title)
    return {"title": title, "conversation": updated}


# ============================================================================
# Admin Endpoints
# ============================================================================

class AdminConfigRequest(BaseModel):
    llm_provider: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_url: Optional[str] = None
    groq_model: Optional[str] = None
    groq_api_key: Optional[str] = None
    gemini_llm_model: Optional[str] = None
    gemini_llm_api_key: Optional[str] = None
    embed_provider: Optional[str] = None
    google_api_key: Optional[str] = None


def _mask_key(key: str) -> str:
    """Devolve uma pré-visualização mascarada: primeiros 6 + '...' + últimos 4.
    Para keys curtas (<= 12 chars) mostra apenas os primeiros 4 + '...'."""
    k = key.strip()
    if not k:
        return ""
    if len(k) <= 12:
        return k[:4] + "..."
    return k[:6] + "..." + k[-4:]


@app.get("/admin/config")
def admin_get_config():
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    gemini_llm_key = (os.getenv("GEMINI_LLM_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")).strip()
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "gemma2:2b"),
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "groq_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        "groq_api_key_set": bool(groq_key),
        "groq_api_key_preview": _mask_key(groq_key),
        "gemini_llm_model": os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
        "gemini_llm_api_key_set": bool(gemini_llm_key),
        "gemini_llm_api_key_preview": _mask_key(gemini_llm_key),
        "embed_provider": os.getenv("EMBED_PROVIDER", "local"),
        "google_api_key_set": bool(google_key),
        "google_api_key_preview": _mask_key(google_key),
    }


@app.post("/admin/config")
def admin_update_config(cfg: AdminConfigRequest):
    """Atualiza variáveis de ambiente em runtime e reinicia o singleton do LLM.
    Campos em branco / None são ignorados (não sobrescrevem o valor atual)."""
    from backend.llm_client import reset_client

    changed: List[str] = []

    if cfg.llm_provider is not None:
        os.environ["LLM_PROVIDER"] = cfg.llm_provider.strip()
        changed.append("LLM_PROVIDER")
    if cfg.ollama_model and cfg.ollama_model.strip():
        os.environ["OLLAMA_MODEL"] = cfg.ollama_model.strip()
        changed.append("OLLAMA_MODEL")
    if cfg.ollama_url and cfg.ollama_url.strip():
        os.environ["OLLAMA_URL"] = cfg.ollama_url.strip()
        changed.append("OLLAMA_URL")
    if cfg.groq_model and cfg.groq_model.strip():
        os.environ["GROQ_MODEL"] = cfg.groq_model.strip()
        changed.append("GROQ_MODEL")
    if cfg.groq_api_key and cfg.groq_api_key.strip():
        os.environ["GROQ_API_KEY"] = cfg.groq_api_key.strip()
        changed.append("GROQ_API_KEY")
    if cfg.gemini_llm_model and cfg.gemini_llm_model.strip():
        os.environ["GEMINI_LLM_MODEL"] = cfg.gemini_llm_model.strip()
        changed.append("GEMINI_LLM_MODEL")
    if cfg.gemini_llm_api_key and cfg.gemini_llm_api_key.strip():
        os.environ["GEMINI_LLM_API_KEY"] = cfg.gemini_llm_api_key.strip()
        changed.append("GEMINI_LLM_API_KEY")
    if cfg.embed_provider is not None:
        os.environ["EMBED_PROVIDER"] = cfg.embed_provider.strip()
        changed.append("EMBED_PROVIDER")
    if cfg.google_api_key and cfg.google_api_key.strip():
        os.environ["GOOGLE_API_KEY"] = cfg.google_api_key.strip()
        changed.append("GOOGLE_API_KEY")

    reset_client()

    return {"updated": changed, "config": admin_get_config()}


# ============================================================================
# Scraper Endpoints
# ============================================================================

class ScraperAddRequest(BaseModel):
    url: str
    label: str = ""
    max_pages: int = 100


def _run_scraper_entry(entry: dict) -> dict:
    url = entry["url"]
    max_pages = max(1, min(int(entry.get("max_pages", 100)), 500))
    run_id = uuid.uuid4().hex[:8]

    # Limpar chunks antigos: por scraper_id (novo) e por source URL (retrocompat)
    for where_filter in [{"scraper_id": entry["id"]}, {"source": url}]:
        try:
            collection.delete(where=where_filter)
        except Exception:
            pass

    try:
        pages = crawl_site(url, max_pages=max_pages)

        all_docs: list = []
        all_meta: list = []
        all_ids: list = []
        for p_idx, page in enumerate(pages):
            for i, chunk in enumerate(chunk_text(page["text"])):
                all_ids.append(f"web-{run_id}-{p_idx}-{i}")
                all_docs.append(chunk)
                all_meta.append({
                    "source":     page["url"],   # URL da página específica
                    "scraper_id": entry["id"],    # para delete limpo na próxima run
                    "seed_url":   url,            # URL raiz do scraper
                    "mode":       "scrape",
                })

        if all_docs:
            collection.add(documents=all_docs, metadatas=all_meta, ids=all_ids)

        entry.update({
            "status":         "ok",
            "last_scraped":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chunks_indexed": len(all_docs),
            "pages_crawled":  len(pages),
            "error":          None,
        })
    except Exception as exc:
        entry.update({
            "status":         "error",
            "last_scraped":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "chunks_indexed": 0,
            "pages_crawled":  0,
            "error":          str(exc),
        })

    return entry


@app.get("/admin/scrapers")
def admin_list_scrapers():
    return _load_scrapers()


@app.post("/admin/scrapers")
def admin_add_scraper(body: ScraperAddRequest):
    url = (body.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL não pode estar vazia.")
    entries = _load_scrapers()
    existing = next((e for e in entries if e["url"] == url), None)
    if existing:
        if body.label and body.label.strip() and body.label.strip() != existing.get("label"):
            existing["label"] = body.label.strip()
            updated = [existing if e["id"] == existing["id"] else e for e in entries]
            _save_scrapers(updated)
        return existing
    entry = {
        "id": uuid.uuid4().hex,
        "url": url,
        "label": (body.label or "").strip() or url,
        "max_pages": max(1, min(body.max_pages, 500)),
        "pages_crawled": 0,
        "last_scraped": None,
        "status": "pending",
        "chunks_indexed": 0,
        "error": None,
    }
    entries.append(entry)
    _save_scrapers(entries)
    return entry


@app.delete("/admin/scrapers/{scraper_id}")
def admin_delete_scraper(scraper_id: str):
    entries = _load_scrapers()
    entry = next((e for e in entries if e["id"] == scraper_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Scraper não encontrado.")
    for where_filter in [{"scraper_id": scraper_id}, {"source": entry["url"]}]:
        try:
            collection.delete(where=where_filter)
        except Exception as exc:
            print(f"[scraper] aviso: falha a apagar chunks ({where_filter}): {exc}")
    updated = [e for e in entries if e["id"] != scraper_id]
    _save_scrapers(updated)
    return {"deleted": scraper_id, "url": entry["url"]}


@app.post("/admin/scrapers/{scraper_id}/run")
def admin_run_scraper(scraper_id: str):
    if not _scraping_enabled():
        raise HTTPException(
            status_code=403,
            detail="O web scraping está desativado. Ative-o primeiro no painel de Administração.",
        )
    entries = _load_scrapers()
    entry = next((e for e in entries if e["id"] == scraper_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Scraper não encontrado.")
    entry = _run_scraper_entry(entry)
    updated = [entry if e["id"] == scraper_id else e for e in entries]
    _save_scrapers(updated)
    return entry


# ============================================================================
# Scraping toggle endpoints
# ============================================================================

@app.get("/admin/scraping/status")
def admin_scraping_status():
    """Estado atual do web scraping (ativado/desativado) e contagem de chunks web."""
    enabled = _scraping_enabled()
    try:
        result = collection.get(where={"mode": "scrape"}, include=[])
        web_chunks = len(result.get("ids") or [])
    except Exception:
        web_chunks = 0
    return {"enabled": enabled, "web_chunks": web_chunks}


@app.post("/admin/scraping/enable")
def admin_scraping_enable():
    """Ativa o web scraping. Os scrapers já configurados podem ser executados manualmente."""
    settings = _load_settings()
    settings["scraping_enabled"] = True
    _save_settings(settings)
    try:
        result = collection.get(where={"mode": "scrape"}, include=[])
        web_chunks = len(result.get("ids") or [])
    except Exception:
        web_chunks = 0
    return {"enabled": True, "web_chunks": web_chunks, "total_chunks": collection.count()}


@app.post("/admin/scraping/disable")
def admin_scraping_disable():
    """Desativa o web scraping e remove TODOS os chunks web do índice.

    Os scrapers continuam configurados mas não podem ser executados enquanto
    o scraping estiver desativado. Os chunks dos PDFs não são afectados.
    """
    try:
        result = collection.get(where={"mode": "scrape"}, include=[])
        removed = len(result.get("ids") or [])
    except Exception:
        removed = 0

    try:
        collection.delete(where={"mode": "scrape"})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao remover chunks web: {exc}")

    settings = _load_settings()
    settings["scraping_enabled"] = False
    _save_settings(settings)

    print(f"[scraping] desativado — {removed} chunks web removidos. Total: {collection.count()}")
    return {
        "enabled": False,
        "removed_chunks": removed,
        "web_chunks": 0,
        "total_chunks": collection.count(),
    }
