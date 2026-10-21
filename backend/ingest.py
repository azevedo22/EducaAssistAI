"""
Extração de conteúdo de PDFs e páginas web.

Três modos de extração de PDFs, em ordem de preferência:

  1. ``extract_markdown_from_pdf`` (PREFERIDO) — PyMuPDF4LLM produz
     Markdown unificado: cabeçalhos preservados, tabelas formatadas como
     markdown, listas mantidas. Ideal para PDFs híbridos (texto + tabelas).
     Resolve o caso típico do calendário académico do IPB onde a prosa
     estruturada (`Início aulas: 16/02/2026`) ficava perdida com a extração
     tabular pura.

  2. ``extract_text_from_pdf`` (FALLBACK) — texto pelo PyMuPDF puro,
     sem layout. Funciona sempre, mesmo se o pymupdf4llm falhar por
     algum motivo (PDF corrompido, encoding raro, etc.).

  3. Extração tabular (em ``pdf_tables.py``) — preserva metadata por linha
     para filtros estruturados (gabinete, dia, departamento). Continua
     útil em paralelo com o markdown como fonte de metadata filtrável.
"""

import os
import re
from collections import deque
from typing import List
from urllib.parse import urljoin, urlparse, urlunparse

import fitz
import requests
from bs4 import BeautifulSoup


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extração base via PyMuPDF: texto cru de todas as páginas, concatenado.

    Sem preservação de layout — usar apenas como fallback. Para conteúdo
    rico (tabelas, headings, listas), preferir ``extract_markdown_from_pdf``.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    with fitz.open(pdf_path) as document:
        pages = [page.get_text() for page in document]

    return "\n\n".join(pages)


def extract_markdown_from_pdf(pdf_path: str) -> str:
    """Extração rica via PyMuPDF4LLM: PDF → Markdown unificado.

    Produz uma string com:
      - cabeçalhos como ``# H1``, ``## H2``, ``### H3``...
      - tabelas como ``| col1 | col2 |``
      - listas como ``- item``
      - parágrafos preservados

    Fallback: se ``pymupdf4llm`` não estiver instalado ou falhar, recorre
    a ``extract_text_from_pdf`` (texto puro). Erros são propagados como
    ``RuntimeError`` com mensagem útil para o caller.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    try:
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError(
            "pymupdf4llm não instalado. Corre `pip install pymupdf4llm` "
            "ou usa extract_text_from_pdf() como fallback."
        ) from exc

    try:
        # to_markdown aceita o caminho ou um doc fitz já aberto.
        # Devolve sempre uma string única (com separadores de página).
        markdown = pymupdf4llm.to_markdown(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Falha a converter PDF para markdown: {exc}") from exc

    if not markdown or not markdown.strip():
        raise RuntimeError(
            f"Markdown vazio para {os.path.basename(pdf_path)} — "
            "PDF pode ser apenas imagens (precisaria de OCR)."
        )

    return markdown


_SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
}

# Tags that are structural noise — removed before extracting content text.
_NOISE_TAGS = [
    "script", "style", "noscript", "iframe", "form", "input",
    "nav", "header", "footer", "aside", "button", "dialog",
]

# Class/id substrings that indicate UI chrome, not content.
_NOISE_PATTERN = re.compile(
    r"cookie|banner|popup|modal|sidebar|breadcrumb|pagination|"
    r"social|share|newsletter|advertisement|widget|language-|"
    r"search|login|register|top-bar|mini-?nav|skip",
    re.IGNORECASE,
)


# ── URL helpers para crawl ────────────────────────────────────────────────────

_SKIP_EXTENSIONS = frozenset([
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".xml", ".rss", ".atom", ".zip", ".gz", ".tar", ".exe", ".pdf",
])


def _get_base_domain(url: str) -> str:
    """www.ipb.pt → ipb.pt,  esa.ipb.pt → ipb.pt"""
    host = urlparse(url).netloc.split(":")[0].lower()
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _normalize_url(url: str) -> str:
    """Remove fragment (#...) e trailing slash para deduplicação."""
    p = urlparse(url)
    return urlunparse(p._replace(fragment="")).rstrip("/")


def _is_crawlable(url: str, base_domain: str) -> bool:
    """True se URL pertence ao domínio base (incluindo *.base_domain) e não é media/ficheiro."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    host = p.netloc.split(":")[0].lower()
    if host != base_domain and not host.endswith("." + base_domain):
        return False
    return not any(p.path.lower().endswith(ext) for ext in _SKIP_EXTENSIONS)


def _extract_links(html: bytes, base_url: str, base_domain: str) -> list:
    """Extrai links válidos do HTML cru (antes de noise removal — links estão em nav/header)."""
    soup = BeautifulSoup(html, "html.parser")
    links: set = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        absolute = _normalize_url(urljoin(base_url, href))
        if _is_crawlable(absolute, base_domain):
            links.add(absolute)
    return list(links)


def _fetch_html(url: str) -> bytes:
    """GET the URL with a browser User-Agent; retries once without SSL verify."""
    try:
        resp = requests.get(url, headers=_SCRAPER_HEADERS, timeout=20,
                            allow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.SSLError:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(url, headers=_SCRAPER_HEADERS, timeout=20,
                            allow_redirects=True, verify=False)
        resp.raise_for_status()
        return resp.content


def _scrape_with_playwright(url: str) -> str:
    """Render a JavaScript SPA with Playwright headless Chromium and extract text.

    Install: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as _PWTimeout
    except ImportError as exc:
        raise RuntimeError(
            "Página SPA detectada mas playwright não está instalado.\n"
            "Para indexar SPAs instala-o:\n"
            "  pip install playwright\n"
            "  playwright install chromium\n"
            "Depois volta a correr este scraper."
        ) from exc

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                try:
                    page.goto(url, timeout=30_000)
                    page.wait_for_load_state("networkidle", timeout=15_000)
                except _PWTimeout:
                    pass
                html = page.content()
            finally:
                browser.close()
    except Exception as exc:
        raise ValueError(f"Erro ao renderizar com Playwright: {exc}") from exc

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for el in list(soup.find_all(True)):
        # After decompose(), BS4 sets attrs=None on the tag and all its
        # descendants. Skip any already-decomposed child that's still in
        # our pre-built list.
        attrs = getattr(el, "attrs", None)
        if attrs is None:
            continue
        cls = " ".join(attrs.get("class") or [])
        el_id = attrs.get("id") or ""
        role = attrs.get("role") or ""
        if (
            _NOISE_PATTERN.search(cls)
            or _NOISE_PATTERN.search(el_id)
            or role in ("navigation", "banner", "contentinfo", "complementary")
        ):
            el.decompose()

    root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find("body")
        or soup
    )
    seen: set = set()
    lines: list = []
    for el in root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6",
         "p", "li", "td", "th", "dt", "dd", "blockquote"]
    ):
        text = re.sub(r"\s+", " ", el.get_text(separator=" ", strip=True))
        if len(text) >= 8 and text not in seen:
            seen.add(text)
            lines.append(text)

    if not lines:
        raise ValueError("Nenhum conteúdo encontrado após renderização JavaScript.")
    return "\n".join(lines)


def _parse_page_text(raw: bytes, url: str) -> str:
    """Parseia HTML cru e devolve texto limpo (sem ruído).

    Detecta SPAs e delega para Playwright se necessário.
    Raises ValueError se a página não tiver conteúdo útil.
    """
    soup = BeautifulSoup(raw, "html.parser")

    # ── SPA detection ──────────────────────────────────────────────────────
    body = soup.find("body")
    if body:
        import copy as _copy
        body_probe = _copy.copy(body)
        for tag in body_probe(["script", "style", "noscript", "link", "meta"]):
            tag.decompose()
        body_text_len = len(body_probe.get_text(strip=True))

        react_root = soup.find(id="root")
        is_spa = (
            (react_root is not None and len(react_root.get_text(strip=True)) < 50)
            or bool(soup.find("app-root") or soup.find(attrs={"ng-version": True}))
            or body_text_len < 100
        )

        if is_spa:
            return _scrape_with_playwright(url)

    # ── Noise removal ──────────────────────────────────────────────────────
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    for el in list(soup.find_all(True)):
        attrs = getattr(el, "attrs", None)
        if attrs is None:
            continue
        cls = " ".join(attrs.get("class") or [])
        el_id = attrs.get("id") or ""
        role = attrs.get("role") or ""
        if (
            _NOISE_PATTERN.search(cls)
            or _NOISE_PATTERN.search(el_id)
            or role in ("navigation", "banner", "contentinfo", "complementary")
        ):
            el.decompose()

    # ── Content extraction ─────────────────────────────────────────────────
    root = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.find("body")
        or soup
    )

    seen: set = set()
    lines: list = []
    for el in root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6",
         "p", "li", "td", "th", "dt", "dd", "blockquote"]
    ):
        text = re.sub(r"\s+", " ", el.get_text(separator=" ", strip=True))
        if len(text) >= 8 and text not in seen:
            seen.add(text)
            lines.append(text)

    if not lines:
        raise ValueError(
            "Nenhum conteúdo encontrado nesta página. "
            "A página pode requerer autenticação ou ter conteúdo apenas em JS."
        )

    return "\n".join(lines)


def scrape_ipb_page(url: str) -> str:
    """Scrape a web page and return deduplicated, noise-free text content.

    Raises:
        ValueError: If the page is a JavaScript SPA shell (empty without JS
            execution) or produces no usable content after noise removal.
        requests.HTTPError: On 4xx/5xx responses.
        requests.Timeout: If the server is unreachable.
    """
    raw = _fetch_html(url)
    return _parse_page_text(raw, url)


def crawl_site(seed_url: str, max_pages: int = 100) -> list:
    """BFS a partir de seed_url, restrito ao mesmo domínio base (*.base_domain).

    Segue todos os links acessíveis dentro do mesmo domínio base (incluindo
    subdomínios). Páginas com erro são ignoradas silenciosamente.

    Returns:
        Lista de dicts {"url": str, "text": str} — uma entrada por página.
    """
    base_domain = _get_base_domain(seed_url)
    visited: set = set()
    queue: deque = deque([_normalize_url(seed_url)])
    results: list = []

    while queue and len(results) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            raw = _fetch_html(url)
        except Exception:
            continue

        # Extrair links do HTML cru — ANTES do noise removal
        # (nav/header com links ainda presentes)
        for link in _extract_links(raw, url, base_domain):
            if link not in visited:
                queue.append(link)

        try:
            text = _parse_page_text(raw, url)
        except Exception:
            continue

        if text:
            results.append({"url": url, "text": text})

    return results


def load_pdf_documents(pdf_paths: List[str]) -> List[dict]:
    documents = []
    for path in pdf_paths:
        text = extract_text_from_pdf(path)
        documents.append({"source": os.path.basename(path), "text": text})
    return documents


def load_text_documents(text_paths: List[str]) -> List[dict]:
    documents = []
    for path in text_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            documents.append({"source": os.path.basename(path), "text": text})
    return documents
