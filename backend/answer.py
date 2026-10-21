"""
Composição da resposta a partir dos chunks recuperados.

Estratégia: enviar os top-k excertos para o LLM (Ollama local ou Groq cloud,
controlado por ``LLM_PROVIDER``) com um prompt estrito que aplica as regras
absolutas do EducaAssistAI:
  1. Responder SÓ com base nos excertos fornecidos.
  2. Se a resposta não estiver nos excertos, devolver uma frase fixa.
  3. Citar a fonte (a citação é propagada pelo retrieval, não pelo LLM).

Se o LLM não estiver disponível, cai para um fallback extrativo (devolve a
passagem mais relevante em bruto) para o sistema continuar a funcionar.

Ver ``backend/llm_client.py`` para a lista completa de variáveis de ambiente
que controlam ambos os fornecedores.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.llm_client import (
    LLMTimeout,
    LLMUnavailable,
    LLMError,
    get_client,
)


# Configuração legacy reexportada para compatibilidade. A configuração
# real está agora em backend.llm_client (um único sítio).
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")

# Limite de caracteres por chunk enviado ao LLM.
# Free tier Groq (20k TPM para 8b-instant): com 10 chunks × 2000 chars
# temos ~5000 tokens de input + ~800 output = ~5800 tok/req → ~3 req/min.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "2000"))

# Máximo de caracteres de histórico incluídos no prompt.
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "2000"))

NO_INFO_MSG = "Não encontrei essa informação nos documentos fornecidos."

# Modelo auxiliar — usado para tarefas rápidas (contextualize_query,
# translate_query) que não exigem raciocínio complexo.
#
# Com Groq: usa llama-3.1-8b-instant (20k TPM) para não esgotar o budget
# do modelo principal. Com Gemini (1M TPM) ou Ollama, None = usa o mesmo
# modelo configurado — sem separação necessária.
def _get_helper_model() -> Optional[str]:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "groq":
        return os.getenv("GROQ_HELPER_MODEL", "llama-3.1-8b-instant")
    return None   # Gemini/Ollama: usa o modelo principal

PROMPT_TEMPLATE = """És um assistente que responde a perguntas sobre documentação \
institucional do Instituto Politécnico de Bragança (IPB), com base \
ESTRITAMENTE nos excertos fornecidos.

REGRAS ABSOLUTAS:
1. Responde APENAS com informação presente nos excertos abaixo. Não uses \
conhecimento externo. Não inventes nomes, datas ou números.
2. Se os excertos não contêm a resposta à pergunta, responde EXATAMENTE: \
"{no_info}"
3. Para perguntas factuais ("quando", "qual", "onde"): responde direto em 1–3 frases.
4. Para perguntas de LISTAGEM ("quem...", "quais...", "que docentes..."): \
enumera TODOS os itens dos excertos que correspondem à pergunta, sem omitir \
nenhum. Usa uma lista com bullet points. Não selecione apenas alguns.
5. APLICA filtros semânticos da pergunta sobre o conjunto recebido. \
Em particular, interpreta tempo: "de manhã" = horários até às 12:00; \
"de tarde" / "à tarde" = horários entre 12:00 e 18:00; \
"à noite" = horários após 18:00. Se a pergunta restringir por hora ("antes das 14h", \
"depois das 16h"), aplica essa restrição e exclui itens que não cumpram. \
Inclui só os itens dos excertos que respeitam TODAS as restrições da pergunta.
6. Lê com atenção o nome dos dias: "segunda-feira", "terça-feira", \
"quarta-feira", "quinta-feira", "sexta-feira", "sábado". Não confundas dias \
diferentes.
7. Responde sempre em português europeu.
8. NO FIM da resposta, escreve uma única linha sozinha com o formato exato:
   FONTES_USADAS: [N], [M], ...
   Onde N e M são os números dos excertos cujo conteúdo usaste para \
construir a resposta. Lista apenas os excertos que efetivamente \
contribuíram para a resposta — NUNCA todos por defeito. Se não usaste \
nenhum (resposta = "{no_info}"), omite esta linha por completo.
   Exemplo: FONTES_USADAS: [1], [3]

EXCERTOS:
{context}

PERGUNTA: {question}

RESPOSTA:"""

# Variante com histórico conversacional — usada quando a conversa já tem
# mensagens anteriores. Acrescenta uma secção HISTÓRICO que o LLM usa para
# interpretar perguntas de seguimento sem repetir informação já dada.
PROMPT_WITH_HISTORY_TEMPLATE = """És um assistente que responde a perguntas sobre documentação \
institucional do Instituto Politécnico de Bragança (IPB), com base \
ESTRITAMENTE nos excertos fornecidos.

REGRAS ABSOLUTAS:
1. Responde APENAS com informação presente nos excertos abaixo. Não uses \
conhecimento externo. Não inventes nomes, datas ou números.
2. Se os excertos não contêm a resposta à pergunta, responde EXATAMENTE: \
"{no_info}"
3. Para perguntas factuais ("quando", "qual", "onde"): responde direto em 1–3 frases.
4. Para perguntas de LISTAGEM ("quem...", "quais...", "que docentes..."): \
enumera TODOS os itens dos excertos que correspondem à pergunta, sem omitir \
nenhum. Usa uma lista com bullet points. Não selecione apenas alguns.
5. APLICA filtros semânticos da pergunta sobre o conjunto recebido. \
Em particular, interpreta tempo: "de manhã" = horários até às 12:00; \
"de tarde" / "à tarde" = horários entre 12:00 e 18:00; \
"à noite" = horários após 18:00. Se a pergunta restringir por hora ("antes das 14h", \
"depois das 16h"), aplica essa restrição e exclui itens que não cumpram. \
Inclui só os itens dos excertos que respeitam TODAS as restrições da pergunta.
6. Lê com atenção o nome dos dias: "segunda-feira", "terça-feira", \
"quarta-feira", "quinta-feira", "sexta-feira", "sábado". Não confundas dias \
diferentes.
7. Responde sempre em português europeu.
8. USA o HISTÓRICO para interpretar perguntas de seguimento \
("e no segundo semestre?", "e à quarta-feira?", "quem mais?", "porquê?"). \
O histórico NÃO é informação documental — é apenas contexto para desambiguar \
a pergunta atual. A resposta deve basear-se SEMPRE nos excertos.
9. NO FIM da resposta, escreve uma única linha sozinha com o formato exato:
   FONTES_USADAS: [N], [M], ...
   Onde N e M são os números dos excertos cujo conteúdo usaste para \
construir a resposta. Lista apenas os excertos que efetivamente \
contribuíram para a resposta — NUNCA todos por defeito. Se não usaste \
nenhum (resposta = "{no_info}"), omite esta linha por completo.
   Exemplo: FONTES_USADAS: [1], [3]

HISTÓRICO DA CONVERSA (mais antiga no topo; usa para interpretar a pergunta atual):
{history}

EXCERTOS:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def _build_history_text(
    messages: List[Dict], max_chars: int = MAX_HISTORY_CHARS
) -> str:
    """Formata as mensagens anteriores como histórico legível para o LLM.

    Itera do mais recente para o mais antigo, acumulando até ``max_chars``,
    depois inverte para apresentar em ordem cronológica. Cada mensagem
    individual é truncada a 800 chars para permitir mais contexto útil.
    """
    labels = {"user": "Utilizador", "assistant": "Assistente"}
    lines: List[str] = []
    total = 0
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role not in labels:
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 800:
            content = content[:800].rstrip() + "…"
        line = f"{labels[role]}: {content}"
        total += len(line) + 1
        if total > max_chars:
            break
        lines.append(line)
    lines.reverse()
    return "\n".join(lines) if lines else ""


# ── Query contextualization ──────────────────────────────────────────────────

_CONTEXTUALIZE_PROMPT = """Dado o histórico de conversa abaixo e a pergunta do utilizador, \
reformula a pergunta para que seja completamente autónoma e compreensível sem o histórico.

REGRAS ABSOLUTAS:
1. Devolves APENAS a pergunta reformulada. NADA MAIS.
2. Se a pergunta já for autónoma (não depende do histórico), devolve-a EXATAMENTE igual.
3. NÃO respondes à pergunta. Apenas reformulas.
4. Mantém a mesma língua (português europeu).
5. Resolve pronomes ("ele", "esse", "aquele"), referências implícitas \
("e à terça?", "quem mais?", "e no segundo semestre?") incorporando o contexto do histório.

HISTÓRICO:
{history}

PERGUNTA: {question}

PERGUNTA REFORMULADA:"""

# Timeout curto: se o LLM demorar mais que isto a reformular, usa a pergunta original.
_CONTEXTUALIZE_TIMEOUT = int(os.getenv("CONTEXTUALIZE_TIMEOUT", "15"))


def contextualize_query(question: str, history: Optional[List[Dict]]) -> str:
    """Reformula uma pergunta de follow-up para ser autónoma.

    Usa o LLM para resolver pronomes e referências ao histórico da conversa,
    de forma a que o retrieval funcione sem contexto conversacional.
    Retorna a pergunta original em caso de falha ou timeout.
    """
    if not history or not question.strip():
        return question

    history_text = _build_history_text(history, max_chars=2000)
    if not history_text:
        return question

    prompt = _CONTEXTUALIZE_PROMPT.format(
        history=history_text,
        question=question.strip(),
    )

    try:
        result = get_client().generate(
            prompt,
            model=_get_helper_model(),   # 8b-instant em Groq; mesmo modelo em Gemini/Ollama
            timeout=_CONTEXTUALIZE_TIMEOUT,
            temperature=0,
            top_p=0.9,
            num_predict=200,
        )
        result = result.strip()
        # Sanidade: se o LLM devolveu algo demasiado longo, provavelmente
        # estava a responder em vez de reformular — usa a original.
        if result and len(result) < len(question) * 5:
            return result
    except Exception as exc:
        print(f"[contextualize] falhou ({exc}); usando pergunta original.")
    return question


# Padrão para detetar a linha "FONTES_USADAS: [1], [3]" no fim da resposta.
# Aceita variações: maiúsculas, espaços, parênteses, vírgulas/espaços.
_USED_SOURCES_RE = re.compile(
    r"^\s*FONTES?_USADAS\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_INDEX_RE = re.compile(r"\[?(\d+)\]?")


def _build_context(hits: List[Dict[str, str]]) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        source = hit.get("source", "fonte desconhecida")
        text = hit.get("text", "").strip()
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rstrip() + "..."
        parts.append(f"[{i}] (fonte: {source})\n{text}")
    return "\n\n".join(parts)


def _collect_sources(hits: List[Dict[str, str]]) -> List[str]:
    sources: List[str] = []
    for hit in hits:
        src = hit.get("source", "documento desconhecido")
        if src not in sources:
            sources.append(src)
    return sources


def _extract_used_sources(
    answer: str, hits: List[Dict[str, str]]
) -> Tuple[str, List[str]]:
    """Procura a linha ``FONTES_USADAS: [1], [3]`` no fim da resposta do LLM
    e devolve ``(answer_sem_essa_linha, fontes_filtradas)``.

    Se o LLM não incluiu a citação, ou se os índices forem inválidos, faz
    fallback para todas as fontes recuperadas (comportamento anterior). O
    objetivo é mostrar ao utilizador apenas os documentos que realmente
    contribuíram para a resposta, em vez do top-k inteiro do retrieval.
    """
    match = _USED_SOURCES_RE.search(answer)
    if not match:
        # LLM não citou — fallback para todas as fontes (comportamento legado).
        return answer, _collect_sources(hits)

    raw = match.group(1)
    indices: List[int] = []
    for m in _INDEX_RE.finditer(raw):
        try:
            i = int(m.group(1)) - 1  # converter para 0-indexado
            if 0 <= i < len(hits):
                indices.append(i)
        except ValueError:
            continue

    if not indices:
        # Linha presente mas sem índices válidos — fallback.
        return _USED_SOURCES_RE.sub("", answer).rstrip(), _collect_sources(hits)

    used: List[str] = []
    for i in indices:
        src = hits[i].get("source", "documento desconhecido")
        if src not in used:
            used.append(src)

    cleaned = _USED_SOURCES_RE.sub("", answer).rstrip()
    return cleaned, used


def _call_llm(prompt: str) -> str:
    """Chamada síntese ao LLM via cliente centralizado (Ollama ou Groq)."""
    return get_client().generate(
        prompt,
        temperature=0,
        top_p=0.9,
        num_predict=int(os.getenv("ANSWER_MAX_TOKENS", "800")),
    )


# Alias de compatibilidade — mantém o nome antigo para callers externos.
_call_ollama = _call_llm


def _extractive_fallback(hits: List[Dict[str, str]]) -> str:
    """Modo de emergência: devolve o excerto mais relevante em bruto."""
    if not hits:
        return NO_INFO_MSG
    excerpt = hits[0].get("text", "").strip().replace("\n", " ")
    if len(excerpt) > 400:
        excerpt = excerpt[:400].rstrip() + "..."
    return (
        "[Modo extrativo — LLM indisponível] Passagem mais relevante encontrada:\n"
        + excerpt
    )


def compose_answer(
    question: str,
    hits: List[Dict[str, str]],
    history: Optional[List[Dict]] = None,
) -> Tuple[str, List[str]]:
    if not hits:
        return NO_INFO_MSG, []

    sources = _collect_sources(hits)
    context_text = _build_context(hits)
    q = question.strip()

    if history:
        history_text = _build_history_text(history)
        prompt = PROMPT_WITH_HISTORY_TEMPLATE.format(
            context=context_text,
            question=q,
            no_info=NO_INFO_MSG,
            history=history_text,
        )
    else:
        prompt = PROMPT_TEMPLATE.format(
            context=context_text,
            question=q,
            no_info=NO_INFO_MSG,
        )

    client = get_client()
    try:
        answer = _call_llm(prompt)
        if not answer:
            answer = _extractive_fallback(hits)
    except LLMUnavailable:
        if client.provider == "groq":
            hint = (
                "(LLM cloud (Groq) inacessível — verifica GROQ_API_KEY e a "
                "ligação à internet.)"
            )
        else:
            hint = (
                f"(Para respostas sintetizadas, instala Ollama e corre "
                f"`ollama pull {client.model}`.)"
            )
        answer = _extractive_fallback(hits) + "\n\n" + hint
    except LLMTimeout:
        answer = (
            _extractive_fallback(hits)
            + f"\n\n(O LLM ({client.provider}) demorou demasiado a responder; "
            "tenta outra vez ou troca de fornecedor via LLM_PROVIDER.)"
        )
    except LLMError as exc:
        answer = _extractive_fallback(hits) + f"\n\n(Erro do LLM: {exc})"
    except Exception as exc:  # pragma: no cover - defensivo
        answer = _extractive_fallback(hits) + f"\n\n(Erro a contactar o LLM: {exc})"

    # Se o LLM reconheceu que não tem informação, não vale a pena listar
    # fontes — a resposta é "não sei".
    if answer.strip() == NO_INFO_MSG:
        return NO_INFO_MSG, []

    # Filtrar fontes: extrair a linha "FONTES_USADAS: [N], [M]" que o LLM
    # acrescenta, e devolver apenas essas fontes (em vez de todo o top-k).
    answer, used_sources = _extract_used_sources(answer, hits)

    # Guarda geral: se o LLM gerou apenas a linha FONTES_USADAS sem texto
    # de resposta (cleaned == ""), tratar como "não encontrei".
    if not answer.strip():
        return NO_INFO_MSG, []

    if not used_sources:
        # Fallback se o parser não conseguiu extrair nada útil.
        used_sources = sources
    return answer, used_sources


# ============================================================================
# Tradutor LLM-driven: pergunta em linguagem natural → filtro ChromaDB JSON.
# ============================================================================

# Timeout dedicado à tradução: queremos uma resposta rápida ou desistir.
TRANSLATE_TIMEOUT = int(os.getenv("OLLAMA_TRANSLATE_TIMEOUT", "30"))

TRANSLATE_PROMPT = """És um tradutor que converte perguntas em linguagem \
natural num filtro estruturado em JSON, para procurar registos numa base de \
dados de documentos do IPB.

CHAVES PESQUISÁVEIS DISPONÍVEIS (e exemplos de valores reais que já lá estão):
{schema}

REGRAS DURAS:
1. Devolves APENAS um objeto JSON válido. Nada de texto antes/depois.
2. Usas EXCLUSIVAMENTE chaves da lista acima. Se a pergunta não referir \
nenhuma dessas chaves, devolves o JSON: null.
3. Para uma única condição: {{"chave": "valor"}} ou {{"chave": true}}.
4. Para várias condições combinadas com E (todas verdadeiras): \
{{"$and": [{{...}}, {{...}}]}}.
5. Para alternativas (qualquer uma): {{"$or": [{{...}}, {{...}}]}}.
6. Para chaves booleanas (começam por has_), o valor é true ou false. \
Para chaves de texto (col_*, gab, dep), o valor é uma STRING.
7. NÃO inventes chaves. NÃO uses operadores que não estejam acima.
8. Filtros como "de manhã", "antes do almoço", "perto do fim do semestre" \
NÃO são estruturáveis — ignora-os no filtro (devolve só os filtros \
estruturáveis, ou null se não houver nenhum).

EXEMPLOS:
Pergunta: "quem atende à sexta-feira?"
JSON: {{"has_sexta": true}}

Pergunta: "que docentes do DCES atendem na sala 13?"
JSON: {{"$and": [{{"dep": "DCES"}}, {{"gab": "13"}}]}}

Pergunta: "quem está na sala 13 e atende à quinta de manhã?"
JSON: {{"$and": [{{"gab": "13"}}, {{"has_quinta": true}}]}}

Pergunta: "qual é o orientador do projeto?"
JSON: null

PERGUNTA: {question}

JSON:"""


# Operadores ChromaDB suportados pelo nosso pipeline
_ALLOWED_OPS = {"$and", "$or"}


def _validate_filter(node: Any, allowed_keys: set) -> Optional[Any]:
    """Valida recursivamente um filtro ChromaDB. Garante que só usa chaves
    da whitelist e operadores permitidos. Devolve a versão validada (com
    cláusulas inválidas removidas) ou None se nada sobrar."""
    if node is None:
        return None
    if not isinstance(node, dict):
        return None

    out: Dict[str, Any] = {}
    for k, v in node.items():
        if k in _ALLOWED_OPS:
            if not isinstance(v, list):
                continue
            children = [_validate_filter(c, allowed_keys) for c in v]
            children = [c for c in children if c]
            if len(children) >= 2:
                out[k] = children
            elif len(children) == 1:
                # Operador unário desnecessário: aplana
                only = children[0]
                if isinstance(only, dict):
                    out.update(only)
        elif k in allowed_keys:
            # Aceita primitivos: str/bool/int/float
            if isinstance(v, (str, bool, int, float)):
                out[k] = v
        # else: chave inválida — silently dropped
    return out if out else None


def _extract_json(text: str) -> Optional[Any]:
    """Extrai o 1º bloco JSON do texto, lidando com prefixos/sufixos do LLM."""
    text = text.strip()
    # Caso a string seja literalmente "null"
    if text.lower() in ("null", "none"):
        return None
    # Procura entre { ... } ou null
    m = re.search(r"\{.*\}|\bnull\b", text, re.DOTALL)
    if not m:
        return None
    snippet = m.group(0)
    if snippet == "null":
        return None
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def translate_query(question: str, schema_text: str, allowed_keys: List[str]) -> Optional[Dict]:
    """Converte uma pergunta NL num filtro ChromaDB válido (ou None).

    Args:
        question: pergunta do utilizador.
        schema_text: descrição das chaves disponíveis (output de
                     schema_store.render_schema_for_prompt).
        allowed_keys: whitelist de nomes de metadata para validação.
    """
    if not schema_text or not allowed_keys:
        return None

    prompt = TRANSLATE_PROMPT.format(
        schema=schema_text, question=question.strip()
    )

    try:
        raw = get_client().generate(
            prompt,
            model=_get_helper_model(),   # 8b-instant em Groq; mesmo modelo em Gemini/Ollama
            timeout=TRANSLATE_TIMEOUT,
            temperature=0,
            top_p=0.9,
            num_predict=128,
            format_json=True,
        )
    except (LLMTimeout, LLMUnavailable, LLMError) as exc:
        print(f"[translate] Falhou: {exc}")
        return None
    except Exception as exc:
        print(f"[translate] Falhou (inesperado): {exc}")
        return None

    parsed = _extract_json(raw)
    if parsed is None:
        return None

    return _validate_filter(parsed, set(allowed_keys))
