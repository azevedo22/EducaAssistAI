"""
Funções de métrica para avaliação quantitativa do pipeline RAG.

Retrieval (nível de documento, após de-duplicação de chunks):
  recall_at_k, precision_at_k  — métricas clássicas de IR
  mrr                          — Mean Reciprocal Rank
  ndcg_at_k                    — Normalized Discounted Cumulative Gain

Generation (comparação predição vs. resposta de referência):
  token_f1   — F1 ao nível de tokens, estilo SQuAD
  bleu       — BLEU-n com brevity penalty (sentence-level)
  rouge_n    — ROUGE-1 / ROUGE-2 (recall de n-gramas)
  rouge_l    — ROUGE-L F1 via Longest Common Subsequence

Normalização:
  normalize_text — lowercase + remoção de acentos + tokenização por \w+

Todas as funções são puras (sem I/O, sem estado global).
"""

import math
import re
import unicodedata
from typing import Dict, List, Sequence, Set


# ── Text normalisation ──────────────────────────────────────────────────────


def normalize_text(text: str) -> List[str]:
    """Tokenize + lowercase + strip accents.

    "Atenção: Hélder às 14h!" → ["atencao", "helder", "as", "14h"]
    Robusto a variações de codificação de acentos entre sistemas.
    """
    text = text.lower()
    # NFD decomposition: "é" → "e" + combining acute; drop combining marks
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"\b\w+\b", text)


# ── Generation metrics ──────────────────────────────────────────────────────


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1: sobreposição bag-of-words entre predição e referência.

    Idêntico à métrica principal do SQuAD:
      F1 = 2 * P * R / (P + R)
    onde P = |pred ∩ ref| / |pred|  e  R = |pred ∩ ref| / |ref|.
    """
    pred = normalize_text(prediction)
    ref = normalize_text(reference)

    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    pred_counts: Dict[str, int] = {}
    for t in pred:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    ref_counts: Dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    common = sum(min(pred_counts.get(t, 0), c) for t, c in ref_counts.items())
    if common == 0:
        return 0.0

    p = common / len(pred)
    r = common / len(ref)
    return 2 * p * r / (p + r)


def _ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    grams: Dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i : i + n])
        grams[g] = grams.get(g, 0) + 1
    return grams


def bleu(prediction: str, reference: str, n: int = 4) -> float:
    """Sentence-level BLEU-n com brevity penalty.

    Média geométrica dos n-gram precisions BLEU-1 ... BLEU-n, multiplicada
    pelo BP = min(1, exp(1 - |ref|/|pred|)).
    Retorna 0.0 se algum n-gram precision for zero (como na implementação
    standard). Para avaliação de frase única, n=1 ou n=2 é mais estável.
    """
    pred_toks = normalize_text(prediction)
    ref_toks = normalize_text(reference)

    if not pred_toks or not ref_toks:
        return 0.0

    log_sum = 0.0
    for k in range(1, n + 1):
        pred_ng = _ngrams(pred_toks, k)
        ref_ng = _ngrams(ref_toks, k)

        if not pred_ng:
            return 0.0

        denom = sum(pred_ng.values())
        matches = sum(min(cnt, ref_ng.get(g, 0)) for g, cnt in pred_ng.items())

        if matches == 0:
            return 0.0

        log_sum += math.log(matches / denom)

    bp = (
        1.0
        if len(pred_toks) >= len(ref_toks)
        else math.exp(1 - len(ref_toks) / len(pred_toks))
    )
    return bp * math.exp(log_sum / n)


def rouge_n(prediction: str, reference: str, n: int = 1) -> float:
    """ROUGE-n recall: fracção dos n-gramas da referência presentes na predição."""
    pred_ng = _ngrams(normalize_text(prediction), n)
    ref_ng = _ngrams(normalize_text(reference), n)

    if not ref_ng:
        return 1.0 if not pred_ng else 0.0

    denom = sum(ref_ng.values())
    matches = sum(min(cnt, pred_ng.get(g, 0)) for g, cnt in ref_ng.items())
    return matches / denom


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 via Longest Common Subsequence (token-level).

    LCS-F1 = 2 * P_lcs * R_lcs / (P_lcs + R_lcs)
    onde P_lcs = LCS / |pred|  e  R_lcs = LCS / |ref|.
    Implementação O(m*n) em espaço O(n) (uma linha de DP de cada vez).
    """
    pred_toks = normalize_text(prediction)
    ref_toks = normalize_text(reference)

    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0

    m, n = len(pred_toks), len(ref_toks)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if pred_toks[i - 1] == ref_toks[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs = prev[n]
    p = lcs / m
    r = lcs / n
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_generation_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Devolve todos os scores de geração para um par (predição, referência)."""
    return {
        "token_f1": token_f1(prediction, reference),
        "bleu1": bleu(prediction, reference, n=1),
        "bleu2": bleu(prediction, reference, n=2),
        "bleu4": bleu(prediction, reference, n=4),
        "rouge1": rouge_n(prediction, reference, n=1),
        "rouge2": rouge_n(prediction, reference, n=2),
        "rouge_l": rouge_l(prediction, reference),
        "exact_match": 1.0
        if normalize_text(prediction) == normalize_text(reference)
        else 0.0,
    }


# ── Retrieval metrics ───────────────────────────────────────────────────────


def _dedup_ordered(sources: Sequence[str]) -> List[str]:
    """Remove duplicados mantendo a ordem de primeira ocorrência."""
    seen: Set[str] = set()
    out: List[str] = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Recall@k: fracção de documentos relevantes encontrada nos top-k únicos.

    `retrieved` é a lista de fontes dos chunks recuperados (com duplicados);
    `relevant`  é o conjunto de documentos esperados.
    A de-duplicação mantém a ordem de retrieval.
    """
    if not relevant:
        return 1.0
    top_k = set(_dedup_ordered(retrieved)[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Precision@k: fracção dos top-k únicos que são relevantes."""
    top_k = _dedup_ordered(retrieved)[:k]
    if not top_k:
        return 0.0
    return sum(1 for s in top_k if s in relevant) / len(top_k)


def mrr(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """MRR: 1 / posição do primeiro documento relevante (de-duplicado)."""
    for i, s in enumerate(_dedup_ordered(retrieved)):
        if s in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """NDCG@k com relevância binária (0/1) por documento único.

    NDCG = DCG / IDCG
    DCG  = Σ rel_i / log2(i+2)   (i=0-indexed, gain=1 se relevante)
    IDCG = DCG com ranking ideal (todos os relevantes no topo)
    """
    top_k = _dedup_ordered(retrieved)[:k]

    def dcg(docs: List[str]) -> float:
        return sum(
            1.0 / math.log2(i + 2) for i, s in enumerate(docs) if s in relevant
        )

    actual = dcg(top_k)
    n_ideal = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))

    if ideal == 0:
        return 1.0 if not relevant else 0.0
    return actual / ideal


def compute_retrieval_metrics(
    retrieved: Sequence[str],
    relevant: Set[str],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Devolve todos os scores de retrieval para um par (retrieved, relevant)."""
    out: Dict[str, float] = {"mrr": mrr(retrieved, relevant)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return out
