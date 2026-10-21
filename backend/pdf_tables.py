"""
Extração inteligente de PDFs tabulares.

Para qualquer PDF com tabela detetável (PyMuPDF.find_tables), produz:
  - 1 chunk de texto por linha, em linguagem natural
  - Metadata GENÉRICA por linha: col_<chave_sanitizada> = valor
  - Metadata ESPECÍFICA (heurísticas de horários): has_<dia>, gab, dep

A metadata genérica permite filtros determinísticos sobre QUALQUER coluna de
QUALQUER PDF, mesmo que o sistema nunca tenha visto aquele tipo de tabela
antes. O caller pode depois fazer collection.get(where={"col_disciplina":
"POO"}) sem saber nada sobre o domínio.
"""

import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import fitz


# ----------------------------------------------------------------------------
# Heurísticas para colunas de dia da semana (apenas para horários do IPB)
# ----------------------------------------------------------------------------
DAY_HEADERS = {
    r"^2[ªa]\.?\s*feira$": "segunda-feira",
    r"^3[ªa]\.?\s*feira$": "terça-feira",
    r"^4[ªa]\.?\s*feira$": "quarta-feira",
    r"^5[ªa]\.?\s*feira$": "quinta-feira",
    r"^6[ªa]\.?\s*feira$": "sexta-feira",
    r"^s[áa]bado$": "sábado",
    r"^domingo$": "domingo",
}
DAY_VALUES = set(DAY_HEADERS.values())
DAY_TO_FLAG = {
    "segunda-feira": "has_segunda",
    "terça-feira":   "has_terca",
    "quarta-feira":  "has_quarta",
    "quinta-feira":  "has_quinta",
    "sexta-feira":   "has_sexta",
    "sábado":        "has_sabado",
    "domingo":       "has_domingo",
}

# Limite de caracteres por valor de metadata (poupa espaço, ChromaDB filtra
# por igualdade exata — valores muito longos não são úteis para filtros).
MAX_META_VALUE_LEN = 200


def _norm(cell) -> str:
    if cell is None:
        return ""
    return " ".join(str(cell).strip().split())


def _normalize_header(h: str) -> str:
    if not h:
        return h
    low = h.strip().lower()
    for pattern, replacement in DAY_HEADERS.items():
        if re.match(pattern, low):
            return replacement
    return h


def _sanitize_key(s: str) -> str:
    """Converte um cabeçalho ('5ª feira', 'DEP.', 'Disciplina') numa chave
    válida para metadata: ascii minúsculo, só alfanumérico+underscore, com
    prefixo 'col_'."""
    if not s:
        return ""
    # Remove acentos
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    # minúsculas, replace não-alfanum por _
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_str.lower()).strip("_")
    if not cleaned:
        return ""
    # Limite razoável de comprimento
    cleaned = cleaned[:40]
    return f"col_{cleaned}"


def _build_row_chunk(header: List[str], cells: List[str]) -> str:
    """Constrói chunk em linguagem natural a partir de uma linha tabular.

    Distingue colunas-identidade (nome, ID, email...) de colunas de horário
    (dias da semana). O nome do dia aparece por extenso para retrieval
    semântico em português.
    """
    identity_parts: List[str] = []
    schedule_parts: List[str] = []
    name = ""

    for h, v in zip(header, cells):
        if not v:
            continue
        h_norm = _normalize_header(h)
        if h_norm in DAY_VALUES:
            schedule_parts.append(f"{h_norm} {v}")
        else:
            if not name:
                name = v
                continue
            if h:
                identity_parts.append(f"{h}: {v}")
            else:
                identity_parts.append(v)

    if not name and not identity_parts and not schedule_parts:
        return ""

    parts: List[str] = []
    if name:
        if identity_parts:
            parts.append(f"{name} ({', '.join(identity_parts)})")
        else:
            parts.append(name)
    elif identity_parts:
        parts.append(" | ".join(identity_parts))

    if schedule_parts:
        parts.append("Horário de atendimento: " + "; ".join(schedule_parts) + ".")

    return ". ".join(p.rstrip(".") for p in parts) + "."


def _build_row_metadata(header: List[str], cells: List[str]) -> Dict:
    """Metadata por linha. Combina:
      (a) col_<sanitizado>: valor — para CADA coluna não vazia (genérico).
      (b) has_<dia> = True — para colunas de dia preenchidas (horários).
      (c) gab/dep — atalhos legíveis para colunas comuns.
    """
    meta: Dict = {}
    for h, v in zip(header, cells):
        if not v:
            continue

        # (a) genérico: chave sanitizada
        key = _sanitize_key(h)
        if key and key not in meta:
            meta[key] = v[:MAX_META_VALUE_LEN]

        # (b) flag por dia (mantém compat. e permite filtros booleanos)
        h_norm = _normalize_header(h)
        flag = DAY_TO_FLAG.get(h_norm)
        if flag:
            meta[flag] = True

        # (c) atalhos legíveis comuns no domínio do IPB
        h_low = (h or "").lower()
        if h_low.startswith("dep") and "dep" not in meta:
            meta["dep"] = v[:MAX_META_VALUE_LEN]
        elif h_low.startswith("gab") and "gab" not in meta:
            meta["gab"] = v[:MAX_META_VALUE_LEN]

    return meta


def extract_table_chunks(pdf_path: str) -> Optional[List[Tuple[str, Dict]]]:
    """
    Devolve lista de (chunk_text, row_metadata) por linha, ou None se não
    detetar tabelas.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    chunks: List[Tuple[str, Dict]] = []

    with fitz.open(pdf_path) as doc:
        try:
            _ = doc[0].find_tables  # noqa: B018
        except AttributeError:
            return None

        last_header: List[str] = []

        for page_num, page in enumerate(doc, start=1):
            try:
                table_finder = page.find_tables()
            except Exception:
                continue

            tables = list(table_finder)
            if not tables:
                continue

            for table in tables:
                try:
                    rows = table.extract()
                except Exception:
                    continue

                if not rows:
                    continue

                header_candidate = [_norm(c) for c in rows[0]]
                # Heurística: se 1ª linha contém email/hora, é dado, não cabeçalho
                looks_like_data = any(
                    re.search(r"@|\d{1,2}:\d{2}", c) for c in header_candidate
                )

                if looks_like_data and last_header:
                    header = last_header
                    data_rows = rows
                else:
                    header = header_candidate
                    last_header = header
                    data_rows = rows[1:]

                for row in data_rows:
                    cells = [_norm(c) for c in row]
                    if not any(cells):
                        continue

                    chunk = _build_row_chunk(header, cells)
                    if chunk and len(chunk) > 5:
                        meta = _build_row_metadata(header, cells)
                        chunks.append((chunk, meta))

    return chunks if chunks else None


def infer_schema(chunks_with_meta: List[Tuple[str, Dict]], max_samples: int = 5) -> Dict:
    """A partir das metadatas das linhas, inferir o esquema do PDF: lista de
    chaves de metadata + amostras de valores. Útil para mostrar ao LLM
    tradutor o que pode filtrar."""
    cols: Dict[str, Dict] = {}
    for _, meta in chunks_with_meta:
        for k, v in meta.items():
            if k not in cols:
                cols[k] = {"key": k, "samples": [], "type": "boolean" if isinstance(v, bool) else "text"}
            if not isinstance(v, bool):
                if v not in cols[k]["samples"] and len(cols[k]["samples"]) < max_samples:
                    cols[k]["samples"].append(v)
    return {
        "row_count": len(chunks_with_meta),
        "columns": list(cols.values()),
    }
