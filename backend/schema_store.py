"""
Registo persistente de esquemas de tabelas por PDF carregado.

Quando o utilizador faz upload de um PDF tabular, registamos aqui que
chaves de metadata existem nesse PDF, com que cabeçalho original, e com
que tipos de valores (amostras). Isto serve de "dicionário de domínio"
para o tradutor LLM converter perguntas em linguagem natural em filtros
estruturados sobre o ChromaDB.

Persistência: um único JSON em backend/db/schemas.json.
"""

import json
import os
import threading
from typing import Dict, List, Optional


_LOCK = threading.Lock()
_SCHEMAS_PATH = os.getenv(
    "SCHEMAS_PATH", os.path.join("backend", "db", "schemas.json")
)


def _load_raw() -> Dict:
    if not os.path.exists(_SCHEMAS_PATH):
        return {}
    try:
        with open(_SCHEMAS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_raw(data: Dict) -> None:
    os.makedirs(os.path.dirname(_SCHEMAS_PATH), exist_ok=True)
    tmp_path = _SCHEMAS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, _SCHEMAS_PATH)


def register_schema(source: str, schema: Dict) -> None:
    """Regista (ou substitui) o esquema de um PDF.

    schema: dict produzido por pdf_tables.infer_schema(),
            no formato {row_count, columns: [{key, samples, type}, ...]}.
    """
    with _LOCK:
        data = _load_raw()
        data[source] = schema
        _save_raw(data)


def remove_schema(source: str) -> None:
    """Remove o esquema de um PDF (chamar quando se elimina o ficheiro)."""
    with _LOCK:
        data = _load_raw()
        if source in data:
            data.pop(source)
            _save_raw(data)


def load_all_schemas() -> Dict[str, Dict]:
    """Devolve todos os esquemas registados, indexados por nome de ficheiro."""
    with _LOCK:
        return _load_raw()


def aggregate_columns() -> List[Dict]:
    """
    Agrega as colunas existentes em TODOS os PDFs registados, para alimentar
    o prompt do tradutor LLM. Para cada chave, junta amostras de todos os
    PDFs em que aparece (até 8 amostras únicas).

    Devolve: [{key, type, samples, sources: [...]}].
    """
    by_key: Dict[str, Dict] = {}
    for source, schema in load_all_schemas().items():
        for col in schema.get("columns", []):
            key = col.get("key")
            if not key:
                continue
            entry = by_key.setdefault(
                key,
                {
                    "key": key,
                    "type": col.get("type", "text"),
                    "samples": [],
                    "sources": [],
                },
            )
            if source not in entry["sources"]:
                entry["sources"].append(source)
            for s in col.get("samples", []):
                if s not in entry["samples"] and len(entry["samples"]) < 8:
                    entry["samples"].append(s)
    return sorted(by_key.values(), key=lambda c: c["key"])


def render_schema_for_prompt(max_keys: int = 15, max_samples: int = 3) -> Optional[str]:
    """Renderiza o esquema agregado em texto para o prompt do tradutor.

    Mantém o prompt curto: por defeito até 15 chaves e 3 amostras por chave —
    suficiente para o LLM encontrar a chave certa, sem inflar o tamanho do
    prompt e o tempo de processamento em CPU.
    """
    cols = aggregate_columns()
    if not cols:
        return None

    lines: List[str] = []
    for col in cols[:max_keys]:
        key = col["key"]
        type_ = col["type"]
        if type_ == "boolean":
            lines.append(f"- {key} (booleano)")
        else:
            samples = col.get("samples", [])
            if samples:
                sample_str = ", ".join(repr(s) for s in samples[:max_samples])
                lines.append(f"- {key} (texto): exemplos {sample_str}")
            else:
                lines.append(f"- {key} (texto)")
    if len(cols) > max_keys:
        lines.append(f"  (... e mais {len(cols) - max_keys} chaves)")
    return "\n".join(lines)
