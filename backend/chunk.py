"""
Chunking de texto para indexação em ChromaDB.

Dois algoritmos disponíveis:

  • ``chunk_text`` — janela deslizante por palavras (overlap configurável).
    Apropriado para texto plano sem estrutura clara. Algoritmo histórico,
    usado em ``mode=text`` e como fallback genérico.

  • ``chunk_markdown`` — divide pelos cabeçalhos do Markdown (``#``, ``##``,
    ``###``), mantendo cada secção como uma unidade semântica. Tabelas e
    listas ficam intactas dentro da sua secção. Secções muito longas são
    sub-divididas com ``chunk_text`` para respeitar o limite de tamanho.
    Usado em ``mode=markdown`` (default das uploads novas).

Os tamanhos são contados em palavras (não caracteres) para ficar próximo do
número de tokens — relevante para o limite do contexto do LLM e para os
limites de input dos modelos de embeddings.
"""

import re
from typing import List


# Limites por defeito. Em palavras (≈ tokens). Sobrescrevíveis nos callers.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200

# Limites para o chunker markdown: secções pequenas mantêm-se intactas;
# secções grandes são partidas. Valores em palavras.
MARKDOWN_TARGET_SIZE = 600    # tamanho-alvo de cada chunk
MARKDOWN_MAX_SIZE = 1200      # máximo absoluto antes de partir
MARKDOWN_MIN_SIZE = 40        # secções mais pequenas são fundidas com a próxima


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """Janela deslizante por palavras com overlap. Apropriado para texto
    plano sem estrutura semântica explícita."""
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# Padrão para detectar uma linha de cabeçalho markdown (# Hx). Ignora
# headers ATX dentro de blocos de código (rara mas possível).
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _split_markdown_sections(markdown: str) -> List[dict]:
    """Divide markdown numa lista de secções, cada uma com:
        { "level": 1..6 ou None, "title": str, "body": str }
    Mantém o texto antes do 1º cabeçalho como secção sem título (level=None)."""
    lines = markdown.splitlines()
    sections: List[dict] = []
    current = {"level": None, "title": "", "body_lines": []}

    in_code_block = False
    for line in lines:
        # Não interpretar #'s dentro de fences ``` ou ~~~
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            current["body_lines"].append(line)
            continue

        if not in_code_block:
            m = _HEADING_RE.match(line)
            if m:
                # Fechar secção anterior
                if current["body_lines"] or current["title"]:
                    current["body"] = "\n".join(current["body_lines"]).strip()
                    sections.append(current)
                current = {
                    "level": len(m.group(1)),
                    "title": m.group(2).strip(),
                    "body_lines": [],
                }
                continue

        current["body_lines"].append(line)

    # Última secção
    current["body"] = "\n".join(current["body_lines"]).strip()
    if current["body"] or current["title"]:
        sections.append(current)

    # Limpar
    for s in sections:
        s.pop("body_lines", None)

    return sections


def _section_word_count(section: dict) -> int:
    title = section.get("title") or ""
    body = section.get("body") or ""
    return len((title + " " + body).split())


def _format_section(section: dict) -> str:
    """Reconstrói uma secção em markdown. Inclui o cabeçalho à frente do
    body para que o chunk preserve contexto (o título é importante para o
    embedding)."""
    title = section.get("title") or ""
    level = section.get("level")
    body = section.get("body") or ""
    if level and title:
        prefix = "#" * level + " " + title
        return (prefix + "\n\n" + body).strip()
    return body.strip()


def chunk_markdown(
    markdown: str,
    target_size: int = MARKDOWN_TARGET_SIZE,
    max_size: int = MARKDOWN_MAX_SIZE,
    min_size: int = MARKDOWN_MIN_SIZE,
    fallback_overlap: int = DEFAULT_OVERLAP,
) -> List[str]:
    """Divide markdown em chunks respeitando a estrutura de cabeçalhos.

    Estratégia:
      1. Parte por cabeçalhos (``# H1``, ``## H2``, ...).
      2. Funde secções demasiado pequenas (< ``min_size`` palavras) com a
         próxima — evita chunks com só "## Título" sem corpo.
      3. Para secções grandes (> ``max_size`` palavras), aplica
         ``chunk_text`` para sub-dividi-las.
      4. Cada chunk começa pelo cabeçalho da sua secção, preservando
         contexto crucial para o embedding.
    """
    if not markdown or not markdown.strip():
        return []

    sections = _split_markdown_sections(markdown)
    if not sections:
        return chunk_text(markdown)

    # Fundir secções pequenas com a próxima (greedy)
    merged: List[dict] = []
    pending: List[dict] = []
    pending_words = 0

    for s in sections:
        wc = _section_word_count(s)
        if pending and (pending_words + wc) <= target_size:
            pending.append(s)
            pending_words += wc
        else:
            if pending:
                merged.append(_combine_sections(pending))
            pending = [s]
            pending_words = wc

    if pending:
        merged.append(_combine_sections(pending))

    # Sub-dividir secções acima do limite máximo
    chunks: List[str] = []
    for s in merged:
        text = _format_section(s)
        wc = len(text.split())
        if wc <= max_size:
            chunks.append(text)
        else:
            # Mantém o título no 1º chunk; nos seguintes, prefixa para contexto
            sub_chunks = chunk_text(text, chunk_size=target_size, overlap=fallback_overlap)
            chunks.extend(sub_chunks)

    # Filtrar chunks vazios ou triviais
    return [c for c in chunks if c and len(c.strip()) > 5]


def _combine_sections(sections: List[dict]) -> dict:
    """Funde várias secções consecutivas numa só para chunking. Cada
    secção é formatada com o seu próprio cabeçalho e o resultado é
    embrulhado num "wrapper" sem cabeçalho próprio (level=None) para
    que ``_format_section`` o devolva tal como está, sem duplicações."""
    if len(sections) == 1:
        return sections[0]
    parts = [_format_section(s) for s in sections]
    return {
        "level": None,
        "title": "",
        "body": "\n\n".join(parts),
    }
