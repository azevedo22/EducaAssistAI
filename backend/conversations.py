"""
Storage de conversas e mensagens via SQLite.

Schema:

  conversations
    id           TEXT PRIMARY KEY     -- uuid4 hex
    title        TEXT NOT NULL        -- "Nova conversa" por defeito
    created_at   TEXT NOT NULL        -- ISO 8601 UTC
    updated_at   TEXT NOT NULL        -- ISO 8601 UTC (timestamp da última mensagem)
    pinned       INTEGER NOT NULL     -- 0/1 (boolean)
    archived     INTEGER NOT NULL     -- 0/1 (soft delete opcional, futuro)

  messages
    id              TEXT PRIMARY KEY
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE
    role            TEXT NOT NULL          -- 'user' | 'assistant' | 'system'
    content         TEXT NOT NULL
    timestamp       TEXT NOT NULL          -- ISO 8601 UTC
    sources         TEXT                   -- JSON: ["file.pdf", ...]
    tools_used      TEXT                   -- JSON: [{"name": "retrieval", "args": {...}}, ...]
    tokens_input    INTEGER                -- tokens da pergunta (estimativa ou real)
    tokens_output   INTEGER                -- tokens da resposta
    agent_actions   TEXT                   -- JSON: [{"action": "...", "result": "..."}, ...]
    execution_state TEXT                   -- 'pending' | 'running' | 'completed' | 'failed'
    metadata        TEXT                   -- JSON: catch-all (modelo usado, provider, etc.)

Indices: por conversation_id (lookups rápidos), updated_at (sort), pinned (filter).

A camada é thread-safe via ``sqlite3.connect(check_same_thread=False)`` e uma
lock partilhada — suficiente para o caso de uso single-process do FastAPI/uvicorn.
"""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Caminho da base de dados. Mesma pasta que o ChromaDB já usa para
# centralizar todos os ficheiros de persistência.
DB_PATH = os.getenv("CONVERSATIONS_DB", os.path.join("backend", "db", "conversations.db"))

# Lock partilhada para writes (SQLite suporta multi-reader/single-writer).
_LOCK = threading.Lock()

# Connection singleton — sqlite3 em modo check_same_thread=False permite
# uso a partir de diferentes threads do uvicorn, mas precisa da lock.
_CONN: Optional[sqlite3.Connection] = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'Nova conversa',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    pinned      INTEGER NOT NULL DEFAULT 0,
    archived    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    sources          TEXT,
    tools_used       TEXT,
    tokens_input     INTEGER,
    tokens_output    INTEGER,
    agent_actions    TEXT,
    execution_state  TEXT,
    metadata         TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id);

CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_pinned
    ON conversations(pinned, updated_at DESC);
"""


# ----------------------------------------------------------------------
# Connection helpers
# ----------------------------------------------------------------------


def _now() -> str:
    """Timestamp ISO 8601 UTC com precisão de segundos."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    """ID curto e legível (uuid4 hex de 16 chars)."""
    return uuid.uuid4().hex[:16]


def get_conn() -> sqlite3.Connection:
    """Devolve a connection singleton, inicializando o schema na 1ª chamada."""
    global _CONN
    if _CONN is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _CONN.execute("PRAGMA foreign_keys = ON;")
        _CONN.executescript(SCHEMA)
        _CONN.commit()
    return _CONN


# ----------------------------------------------------------------------
# Serialização de campos JSON
# ----------------------------------------------------------------------


def _row_to_message(row: sqlite3.Row) -> Dict[str, Any]:
    """Converte uma linha da tabela messages num dict com os campos JSON
    descodificados. Campos NULL devolvem o default sensato (lista vazia, etc.)."""
    if row is None:
        return {}

    def _safe_json(val, default):
        if not val:
            return default
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return default

    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "role": row["role"],
        "content": row["content"],
        "timestamp": row["timestamp"],
        "sources": _safe_json(row["sources"], []),
        "tools_used": _safe_json(row["tools_used"], []),
        "tokens_input": row["tokens_input"],
        "tokens_output": row["tokens_output"],
        "agent_actions": _safe_json(row["agent_actions"], []),
        "execution_state": row["execution_state"],
        "metadata": _safe_json(row["metadata"], {}),
    }


def _row_to_conversation(row: sqlite3.Row, message_count: Optional[int] = None) -> Dict[str, Any]:
    if row is None:
        return {}
    out = {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "pinned": bool(row["pinned"]),
        "archived": bool(row["archived"]),
    }
    if message_count is not None:
        out["message_count"] = message_count
    return out


def _json_or_null(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value  # já é JSON serializado
    return json.dumps(value, ensure_ascii=False)


# ----------------------------------------------------------------------
# Conversations CRUD
# ----------------------------------------------------------------------


def create_conversation(title: Optional[str] = None) -> Dict[str, Any]:
    """Cria uma conversa nova vazia e devolve-a."""
    conv_id = _new_id()
    now = _now()
    with _LOCK:
        conn = get_conn()
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, pinned, archived) "
            "VALUES (?, ?, ?, ?, 0, 0)",
            (conv_id, title or "Nova conversa", now, now),
        )
        conn.commit()
    return get_conversation(conv_id, include_messages=False) or {}


def list_conversations(include_archived: bool = False) -> List[Dict[str, Any]]:
    """Lista conversas, ordenadas por pinned (fixadas no topo) depois updated_at desc.

    Inclui ``message_count`` por conversa via subquery (1 round-trip)."""
    conn = get_conn()
    where = "" if include_archived else "WHERE c.archived = 0"
    rows = conn.execute(
        f"""
        SELECT c.*,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS msg_count
        FROM conversations c
        {where}
        ORDER BY c.pinned DESC, c.updated_at DESC
        """
    ).fetchall()
    return [_row_to_conversation(r, message_count=r["msg_count"]) for r in rows]


def get_conversation(
    conversation_id: str, include_messages: bool = True
) -> Optional[Dict[str, Any]]:
    """Devolve uma conversa, opcionalmente com as suas mensagens (ordem
    cronológica). Devolve None se não existir."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    if row is None:
        return None

    msg_count_row = conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()

    conv = _row_to_conversation(row, message_count=msg_count_row["n"])

    if include_messages:
        msg_rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
            (conversation_id,),
        ).fetchall()
        conv["messages"] = [_row_to_message(r) for r in msg_rows]

    return conv


def update_conversation(
    conversation_id: str,
    title: Optional[str] = None,
    pinned: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Atualização parcial: só os campos não-None são tocados."""
    fields: List[str] = []
    values: List[Any] = []
    if title is not None:
        fields.append("title = ?")
        values.append(title)
    if pinned is not None:
        fields.append("pinned = ?")
        values.append(1 if pinned else 0)
    if archived is not None:
        fields.append("archived = ?")
        values.append(1 if archived else 0)

    if not fields:
        # No-op idempotente
        return get_conversation(conversation_id, include_messages=False)

    values.append(conversation_id)
    with _LOCK:
        conn = get_conn()
        cursor = conn.execute(
            f"UPDATE conversations SET {', '.join(fields)} WHERE id = ?", values
        )
        if cursor.rowcount == 0:
            return None
        conn.commit()

    return get_conversation(conversation_id, include_messages=False)


def delete_conversation(conversation_id: str) -> bool:
    """Apaga conversa + mensagens (cascade). Devolve True se algo foi apagado."""
    with _LOCK:
        conn = get_conn()
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


# ----------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    sources: Optional[List[str]] = None,
    tools_used: Optional[List[Dict[str, Any]]] = None,
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
    agent_actions: Optional[List[Dict[str, Any]]] = None,
    execution_state: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Adiciona uma mensagem a uma conversa e atualiza ``updated_at`` da
    conversa para refletir a atividade recente (importante para ordenação
    e agrupamento por tempo)."""
    if role not in ("user", "assistant", "system"):
        raise ValueError(f"role inválido: {role!r}. Use 'user', 'assistant' ou 'system'.")

    msg_id = _new_id()
    now = _now()
    with _LOCK:
        conn = get_conn()
        # Verifica que a conversa existe
        conv = conn.execute(
            "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if conv is None:
            return None

        conn.execute(
            """
            INSERT INTO messages (
                id, conversation_id, role, content, timestamp,
                sources, tools_used, tokens_input, tokens_output,
                agent_actions, execution_state, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg_id,
                conversation_id,
                role,
                content,
                now,
                _json_or_null(sources),
                _json_or_null(tools_used),
                tokens_input,
                tokens_output,
                _json_or_null(agent_actions),
                execution_state,
                _json_or_null(metadata),
            ),
        )
        # Atualizar timestamp da conversa
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        conn.commit()

    return get_message(msg_id)


def get_message(message_id: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM messages WHERE id = ?", (message_id,)
    ).fetchone()
    return _row_to_message(row) if row else None


def list_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """Devolve todas as mensagens de uma conversa em ordem cronológica."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC",
        (conversation_id,),
    ).fetchall()
    return [_row_to_message(r) for r in rows]


def get_recent_messages(conversation_id: str, n: int = 10) -> List[Dict[str, Any]]:
    """Devolve as últimas ``n`` mensagens de uma conversa em ordem cronológica.

    Mais eficiente que ``list_messages`` para contexto conversacional —
    usa ORDER BY … DESC LIMIT n e inverte o resultado, evitando carregar
    o histórico completo para conversas longas.
    """
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT * FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp DESC, rowid DESC
        LIMIT ?
        """,
        (conversation_id, n),
    ).fetchall()
    # Inverter para ordem cronológica (mais antiga primeiro).
    return [_row_to_message(r) for r in reversed(rows)]


def delete_message(message_id: str) -> bool:
    with _LOCK:
        conn = get_conn()
        cursor = conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        conn.commit()
        return cursor.rowcount > 0


# ----------------------------------------------------------------------
# Pesquisa
# ----------------------------------------------------------------------


def _make_snippet(content: str, query: str, padding: int = 50) -> str:
    """Constrói um snippet curto que mostra o contexto à volta do match.

    Devolve até ``len(query) + 2*padding`` caracteres centrados no match,
    com ``...`` à frente/atrás se foi truncado. Insensível a maiúsculas."""
    if not content or not query:
        return content[:120] if content else ""

    idx = content.lower().find(query.lower())
    if idx < 0:
        # Match foi no título, não no conteúdo: devolve o início.
        return (content[:padding * 2] + "...") if len(content) > padding * 2 else content

    start = max(0, idx - padding)
    end = min(len(content), idx + len(query) + padding)
    snippet = content[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    # Substituir múltiplos whitespaces por espaço único para snippets compactos.
    return " ".join(snippet.split())


def search_conversations(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Procura conversas cujo título OU alguma mensagem contém ``query``.

    Para cada conversa devolvida inclui um campo ``match`` com:
      - ``where``: ``'title'`` ou ``'message'`` (onde a primeira coincidência ocorreu)
      - ``snippet``: contexto à volta da palavra encontrada
      - ``message_id``: ID da mensagem (se match foi em mensagem)

    Implementação: simples ``LIKE %query%`` com case-insensitive (SQLite
    default em LIKE com BINARY collation seria case-sensitive, mas usamos
    LOWER() para uniformizar). Para corpus muito grandes (>10k conversas)
    podemos migrar para FTS5 — pendente para uma fase futura.
    """
    q = (query or "").strip()
    if not q:
        return []

    pattern = f"%{q.lower()}%"
    conn = get_conn()

    rows = conn.execute(
        """
        SELECT DISTINCT c.*,
               (SELECT COUNT(*) FROM messages m2 WHERE m2.conversation_id = c.id) AS msg_count
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE LOWER(c.title) LIKE ?
           OR LOWER(m.content) LIKE ?
        ORDER BY c.pinned DESC, c.updated_at DESC
        LIMIT ?
        """,
        (pattern, pattern, limit),
    ).fetchall()

    results: List[Dict[str, Any]] = []
    for r in rows:
        conv = _row_to_conversation(r, message_count=r["msg_count"])

        match_info: Dict[str, Any] = {}
        # 1) Match no título tem prioridade
        if q.lower() in (r["title"] or "").lower():
            match_info = {
                "where": "title",
                "snippet": _make_snippet(r["title"], q),
            }
        else:
            # 2) Procurar a 1ª mensagem com match (cronologicamente)
            msg = conn.execute(
                """
                SELECT id, content FROM messages
                WHERE conversation_id = ? AND LOWER(content) LIKE ?
                ORDER BY timestamp ASC LIMIT 1
                """,
                (conv["id"], pattern),
            ).fetchone()
            if msg:
                match_info = {
                    "where": "message",
                    "snippet": _make_snippet(msg["content"], q),
                    "message_id": msg["id"],
                }

        conv["match"] = match_info
        results.append(conv)

    return results


# ----------------------------------------------------------------------
# Helpers — agrupamento por tempo (usado pelo frontend Fase 3)
# ----------------------------------------------------------------------


def group_conversations_by_time(
    conversations: List[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Agrupa uma lista de conversas em buckets temporais para o sidebar.

    Buckets (chaves): ``pinned``, ``today``, ``yesterday``, ``last_7_days``,
    ``last_30_days``, ``older``. Pinned é independente do tempo (fixadas
    aparecem sempre no topo)."""
    if now is None:
        now = datetime.now(tz=timezone.utc)

    today = now.date()
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "pinned": [],
        "today": [],
        "yesterday": [],
        "last_7_days": [],
        "last_30_days": [],
        "older": [],
    }

    for c in conversations:
        if c.get("pinned"):
            buckets["pinned"].append(c)
            continue

        try:
            updated = datetime.fromisoformat(c["updated_at"])
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            buckets["older"].append(c)
            continue

        delta = (today - updated.date()).days
        if delta <= 0:
            buckets["today"].append(c)
        elif delta == 1:
            buckets["yesterday"].append(c)
        elif delta <= 7:
            buckets["last_7_days"].append(c)
        elif delta <= 30:
            buckets["last_30_days"].append(c)
        else:
            buckets["older"].append(c)

    return buckets
