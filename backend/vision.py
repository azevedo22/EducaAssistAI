"""
Extração multimodal via Gemini Vision (gemini-1.5-flash).

Processa cada página de um PDF como imagem e pede ao LLM que extraia
todo o conteúdo visível: texto estruturado, tabelas (→ Markdown),
diagramas e figuras anotadas.

Casos de uso principais:
  • PDFs digitalizados (scanned) — sem camada de texto, só imagens
  • Tabelas complexas que o PyMuPDF renderiza mal
  • Diagramas com texto incorporado

Requisito: GOOGLE_API_KEY definida no ambiente.
"""

import os
import time
from typing import List, Optional

import fitz  # PyMuPDF já é dependência do projeto


VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
VISION_DPI   = int(os.getenv("VISION_DPI", "150"))
# Delay entre páginas para respeitar o tier gratuito (15 RPM).
# Com 0.5s entre páginas nunca excede 10 req/min com folga.
VISION_PAGE_DELAY = float(os.getenv("VISION_PAGE_DELAY", "4.5"))

_EXTRACT_PROMPT = (
    "Analisa esta página de um documento institucional e extrai TODO o conteúdo "
    "de forma estruturada.\n\n"
    "REGRAS:\n"
    "1. Extrai TODO o texto visível, preservando hierarquia: # para títulos "
    "principais, ## para subtítulos, ### para secções.\n"
    "2. Para tabelas: reproduz como tabela Markdown (| col1 | col2 | ... |), "
    "incluindo cabeçalhos e linha de separação.\n"
    "3. Para diagramas, fluxogramas ou figuras com texto: descreve o conteúdo "
    "relevante em prosa clara.\n"
    "4. Mantém a ordem lógica de leitura (esquerda→direita, topo→fundo).\n"
    "5. Não adiciones comentários sobre a qualidade da imagem.\n"
    "6. Se a página estiver em branco ou for apenas decoração, responde: [VAZIO]\n\n"
    "Devolve apenas o conteúdo extraído, sem preâmbulo."
)


def _page_to_png_bytes(page: fitz.Page, dpi: int = VISION_DPI) -> bytes:
    """Renderiza uma página PDF como PNG em memória."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("png")


def is_scanned_pdf(pdf_path: str, text_threshold: int = 80) -> bool:
    """Heurística: PDF provavelmente digitalizado se tiver poucas chars por
    página mas contiver imagens incorporadas.

    threshold=80 chars/página cobre PDFs onde PyMuPDF extrai apenas
    artefactos (números de página, cabeçalhos vazios) mas não texto real.
    """
    try:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                return False
            total_text = sum(len(page.get_text().strip()) for page in doc)
            has_images = any(page.get_images() for page in doc)
            avg_chars = total_text / len(doc)
            return avg_chars < text_threshold and has_images
    except Exception:
        return False


def extract_with_vision(
    pdf_path: str,
    api_key: Optional[str] = None,
    model: str = VISION_MODEL,
    dpi: int = VISION_DPI,
    page_delay: float = VISION_PAGE_DELAY,
) -> str:
    """Extrai texto de um PDF enviando cada página como imagem ao Gemini.

    Devolve uma string Markdown com secções por página.

    Raises:
        FileNotFoundError: PDF não existe.
        RuntimeError: GOOGLE_API_KEY ausente, pacote não instalado, ou
                      nenhum conteúdo extraído após processar todas as páginas.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai não instalado. "
            "Corre: pip install google-genai"
        ) from exc

    key = (api_key or os.getenv("GOOGLE_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY não definida. "
            "Necessária para extração visual com Gemini."
        )

    client = genai.Client(api_key=key)
    gen_config = genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=2048,
    )

    pages_text: List[str] = []

    with fitz.open(pdf_path) as doc:
        n_pages = len(doc)
        print(f"[vision] A processar {n_pages} página(s) com {model}…")

        for page_num, page in enumerate(doc, start=1):
            png_bytes = _page_to_png_bytes(page, dpi=dpi)
            image_part = genai_types.Part.from_bytes(
                data=png_bytes, mime_type="image/png"
            )

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[_EXTRACT_PROMPT, image_part],
                    config=gen_config,
                )
                text = response.text.strip()
                if text and text != "[VAZIO]":
                    pages_text.append(f"## Página {page_num}\n\n{text}")
                    print(f"[vision] Página {page_num}/{n_pages}: {len(text)} chars extraídos")
                else:
                    print(f"[vision] Página {page_num}/{n_pages}: vazia/decoração, ignorada")
            except Exception as exc:
                print(f"[vision] Página {page_num}/{n_pages} falhou: {exc}")

            if page_delay > 0 and page_num < n_pages:
                time.sleep(page_delay)

    if not pages_text:
        raise RuntimeError(
            "Nenhum conteúdo extraído via Gemini Vision. "
            "Verifica a GOOGLE_API_KEY e o conteúdo do PDF."
        )

    return "\n\n".join(pages_text)
