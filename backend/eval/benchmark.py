"""
Benchmark rigoroso de avaliação do EducaAssistAI.

Mede duas dimensões independentes do pipeline RAG:

  RETRIEVAL  (via POST /retrieve)
    Recall@k      k = 1, 3, 5, 10  — fracção de docs relevantes encontrada
    Precision@k   k = 1, 3, 5      — fracção dos top-k que são relevantes
    MRR                            — Mean Reciprocal Rank
    NDCG@k        k = 5, 10        — Normalized Discounted Cumulative Gain

  GENERATION  (via POST /query)
    Token F1                       — sobreposição bag-of-tokens (estilo SQuAD)
    BLEU-1 / BLEU-2 / BLEU-4       — n-gram precision com brevity penalty
    ROUGE-1 / ROUGE-2 / ROUGE-L    — recall de n-gramas / LCS F1
    Source recall / precision       — fontes citadas vs. fontes esperadas

USO:
  # Backend em execução em localhost:8000
  python -m backend.eval.benchmark

  # Gravar resultados detalhados em JSON
  python -m backend.eval.benchmark --out resultados.json

  # Só métricas de retrieval (sem LLM; muito mais rápido)
  python -m backend.eval.benchmark --retrieval-only

  # Só um subconjunto de perguntas
  python -m backend.eval.benchmark --filter fact_

  # k's customizados para retrieval
  python -m backend.eval.benchmark --k 1 5 10

REQUISITOS:
  backend a correr: uvicorn backend.app:app --reload
  gold_set_v2.json no mesmo directório que este ficheiro

CONTEXTO DA LITERATURA RAG (comparação na secção final do relatório):
  Lewis et al. 2020  — RAG original; NaturalQuestions (inglês)
  Es et al. 2023     — RAGAS: avaliação automática de RAG
  Shi et al. 2023    — Lost in the Middle (posição dos chunks)
  Os nossos números são em português e num corpus de domínio específico
  — comparações directas são aproximadas, servem como referência de ordem
  de grandeza.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ks always computed so the summary table has complete rows regardless of --k
_TABLE_KS: Set[int] = {1, 3, 5, 10}

import requests

from backend.eval.metrics import (
    compute_generation_metrics,
    compute_retrieval_metrics,
)


# ── HTTP helpers ─────────────────────────────────────────────────────────────

BASE_TIMEOUT = 300  # segundos por query


def _post(url: str, path: str, payload: Dict, timeout: int) -> Dict:
    try:
        t0 = time.time()
        r = requests.post(f"{url.rstrip('/')}{path}", json=payload, timeout=timeout)
        elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()
        data["_elapsed_s"] = elapsed
        data["_ok"] = True
        return data
    except requests.exceptions.Timeout:
        return {"_ok": False, "_error": f"timeout após {timeout}s", "_elapsed_s": timeout}
    except Exception as exc:
        return {"_ok": False, "_error": str(exc), "_elapsed_s": 0.0}


def retrieve(url: str, question: str, k: int, timeout: int) -> Dict:
    """POST /retrieve → lista de chunks ordenada por relevância."""
    return _post(url, "/retrieve", {"question": question, "k": k}, timeout)


def query(url: str, question: str, timeout: int) -> Dict:
    """POST /query → resposta sintetizada + fontes citadas."""
    return _post(url, "/query", {"question": question}, timeout)


def health_check(url: str) -> bool:
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_retrieval(test: Dict, retrieve_resp: Dict, ks: List[int]) -> Dict[str, float]:
    """Métricas de retrieval para uma pergunta.

    Sempre computa para _TABLE_KS ∪ ks para que as colunas do relatório
    tenham valores mesmo quando o utilizador passa --k com um subconjunto.
    """
    all_ks = sorted(set(ks) | _TABLE_KS)

    if not retrieve_resp.get("_ok"):
        return {f"recall@{k}": 0.0 for k in all_ks} | {
            f"precision@{k}": 0.0 for k in all_ks
        } | {"mrr": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}

    chunks = retrieve_resp.get("chunks", [])
    retrieved_sources: List[str] = [c.get("source", "") for c in chunks]
    expected: Set[str] = set(test.get("expected_sources", []))

    # Para categorias sem fontes esperadas, retrieval é N/A.
    # Retornamos 1.0 para não penalizar o sistema injustamente;
    # a avaliação real destas categorias é no passo de generation.
    if not expected and test.get("category") in ("unknown", "anti_hallucination"):
        return {f"recall@{k}": 1.0 for k in all_ks} | {
            f"precision@{k}": 1.0 for k in all_ks
        } | {"mrr": 1.0, "ndcg@5": 1.0, "ndcg@10": 1.0}

    return compute_retrieval_metrics(retrieved_sources, expected, ks=all_ks)


def score_generation(test: Dict, query_resp: Dict) -> Tuple[Dict[str, float], List[str]]:
    """Métricas de geração para uma pergunta. Retorna (scores, notes)."""
    notes: List[str] = []

    if not query_resp.get("_ok"):
        notes.append(f"backend error: {query_resp.get('_error', '?')}")
        zero = {k: 0.0 for k in (
            "token_f1", "bleu1", "bleu2", "bleu4",
            "rouge1", "rouge2", "rouge_l", "exact_match",
            "source_recall", "source_precision",
        )}
        return zero, notes

    answer: str = query_resp.get("answer", "")
    cited: Set[str] = set(query_resp.get("sources", []))
    expected: Set[str] = set(test.get("expected_sources", []))
    reference: str = test.get("reference_answer", "")

    gen = compute_generation_metrics(answer, reference) if reference else {}

    # Source-level scores
    if test.get("category") in ("unknown", "anti_hallucination"):
        # Deve citar ZERO fontes
        source_recall = 1.0 if not cited else 0.0
        source_precision = 1.0 if not cited else 0.0
        if cited:
            notes.append(f"fontes inesperadas para {test['category']}: {cited}")
    elif expected:
        intersection = cited & expected
        source_recall = len(intersection) / len(expected)
        source_precision = len(intersection) / len(cited) if cited else 0.0
        if source_recall < 1.0:
            notes.append(f"fontes em falta: {expected - cited}")
    else:
        source_recall = 1.0
        source_precision = 1.0

    scores = dict(gen)
    scores["source_recall"] = source_recall
    scores["source_precision"] = source_precision

    # Notas de correctness legacy (compatibilidade com run.py)
    must_contain = test.get("must_contain", [])
    if must_contain:
        al = answer.lower()
        missed = [kw for kw in must_contain if kw.lower() not in al]
        if missed:
            notes.append(f"keywords em falta: {missed}")

    expected_phrase = test.get("expected_answer_phrase")
    if expected_phrase and expected_phrase.lower() not in answer.lower():
        notes.append(f"frase esperada ausente: {expected_phrase!r}")

    return scores, notes


# ── Formatting ────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v:.1%}"


def print_test_row(
    test: Dict,
    retr: Optional[Dict[str, float]],
    gen: Optional[Dict[str, float]],
    notes: List[str],
    elapsed_s: float,
) -> None:
    qid = test["id"]
    cat = test.get("category", "—")[:8]
    q = test["question"][:60]
    print(f"\n  [{qid:<10}] ({cat:<8}) {q}")

    if retr:
        r1  = retr.get("recall@1", 0.0)
        r5  = retr.get("recall@5", 0.0)
        r10 = retr.get("recall@10", 0.0)
        mrr = retr.get("mrr", 0.0)
        nd5 = retr.get("ndcg@5", 0.0)
        print(
            f"    Retrieval  R@1={_pct(r1)}  R@5={_pct(r5)}  R@10={_pct(r10)}"
            f"  MRR={_pct(mrr)}  NDCG@5={_pct(nd5)}"
        )

    if gen:
        f1   = gen.get("token_f1", 0.0)
        r1   = gen.get("rouge1", 0.0)
        rl   = gen.get("rouge_l", 0.0)
        b1   = gen.get("bleu1", 0.0)
        sr   = gen.get("source_recall", 0.0)
        sp   = gen.get("source_precision", 0.0)
        print(
            f"    Generation F1={_pct(f1)}  R1={_pct(r1)}  RL={_pct(rl)}"
            f"  B1={_pct(b1)}  SrcRec={_pct(sr)}  SrcPrec={_pct(sp)}"
            f"  ({elapsed_s:.1f}s)"
        )

    for note in notes[:3]:
        print(f"    · {note}")


def print_summary(results: List[Dict], ks: List[int], retrieval_only: bool) -> None:
    n = len(results)
    if n == 0:
        print("  Nenhum teste executado.")
        return

    def avg_retr(key: str) -> float:
        vals = [r["retrieval"].get(key, 0.0) for r in results if r.get("retrieval")]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_gen(key: str) -> float:
        vals = [r["generation"].get(key, 0.0) for r in results if r.get("generation")]
        return sum(vals) / len(vals) if vals else 0.0

    avg_elapsed = sum(r.get("elapsed_query_s", 0.0) for r in results) / n

    sep = "═" * 84
    print()
    print(sep)
    print("  BENCHMARK EducaAssistAI — RESULTADOS GLOBAIS")
    print(sep)
    print(f"  Total de testes : {n}")
    print(f"  Latência média  : {avg_elapsed:.1f}s / query")

    # ── Retrieval table ──
    print()
    print("  ── RETRIEVAL ──────────────────────────────────────────────────────")
    row_fmt = "  {:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
    print(row_fmt.format("", "R@1", "R@3", "R@5", "R@10", "P@1", "MRR", "NDCG@5"))
    print("  " + "─" * 70)

    # Por categoria
    cats: Dict[str, List] = {}
    for r in results:
        cats.setdefault(r["test"].get("category", "?"), []).append(r)

    for cat in sorted(cats):
        rs = cats[cat]
        def ca(key: str, _rs: List = rs) -> float:
            return sum(x["retrieval"].get(key, 0.0) for x in _rs if x.get("retrieval")) / len(_rs)
        print(row_fmt.format(
            cat[:14],
            _pct(ca("recall@1")), _pct(ca("recall@3")), _pct(ca("recall@5")), _pct(ca("recall@10")),
            _pct(ca("precision@1")), _pct(ca("mrr")), _pct(ca("ndcg@5")),
        ))

    print("  " + "─" * 70)
    print(row_fmt.format(
        "GLOBAL",
        _pct(avg_retr("recall@1")), _pct(avg_retr("recall@3")),
        _pct(avg_retr("recall@5")), _pct(avg_retr("recall@10")),
        _pct(avg_retr("precision@1")), _pct(avg_retr("mrr")), _pct(avg_retr("ndcg@5")),
    ))
    print(f"  {'':14} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'P@1':>8} {'MRR':>8} {'NDCG@5':>8}")

    if not retrieval_only:
        # ── Generation table ──
        print()
        print("  ── GENERATION ─────────────────────────────────────────────────────")
        gen_fmt = "  {:<14} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
        print(gen_fmt.format("", "F1", "ROUGE-1", "ROUGE-L", "BLEU-1", "BLEU-2", "SrcRec", "SrcPrec"))
        print("  " + "─" * 70)

        for cat in sorted(cats):
            rs = cats[cat]
            def cg(key: str, _rs: List = rs) -> float:
                return sum(x["generation"].get(key, 0.0) for x in _rs if x.get("generation")) / len(_rs)
            print(gen_fmt.format(
                cat[:14],
                _pct(cg("token_f1")), _pct(cg("rouge1")), _pct(cg("rouge_l")),
                _pct(cg("bleu1")), _pct(cg("bleu2")),
                _pct(cg("source_recall")), _pct(cg("source_precision")),
            ))

        print("  " + "─" * 70)
        print(gen_fmt.format(
            "GLOBAL",
            _pct(avg_gen("token_f1")), _pct(avg_gen("rouge1")), _pct(avg_gen("rouge_l")),
            _pct(avg_gen("bleu1")), _pct(avg_gen("bleu2")),
            _pct(avg_gen("source_recall")), _pct(avg_gen("source_precision")),
        ))

    # ── Falhas ──
    low = [
        r for r in results
        if (r.get("retrieval", {}).get("recall@5", 1.0) < 0.5
            or r.get("generation", {}).get("token_f1", 1.0) < 0.3)
    ]
    if low:
        print()
        print(f"  Casos com performance baixa (Recall@5<50% ou F1<30%) — {len(low)} testes:")
        for r in low:
            qid = r["test"]["id"]
            cat = r["test"].get("category", "—")
            q = r["test"]["question"][:60]
            r5 = r.get("retrieval", {}).get("recall@5", 0.0)
            f1 = r.get("generation", {}).get("token_f1", 0.0)
            print(f"    · [{qid}] ({cat}) {q}")
            print(f"      R@5={_pct(r5)}  F1={_pct(f1)}")

    # ── Contexto da literatura ──
    print()
    print("  ── CONTEXTO RAG (literatura) ──────────────────────────────────────")
    na = "    N/A   "
    our_r5 = _pct(avg_retr("recall@5"))
    our_f1 = _pct(avg_gen("token_f1")) if not retrieval_only else na
    our_r1 = _pct(avg_gen("rouge1"))   if not retrieval_only else na

    print()
    print("  Os números abaixo são de benchmarks publicados e servem como")
    print("  referência de grandeza. PT-PT / domínio específico — comparações")
    print("  directas são aproximadas.")
    print()
    print(f"  {'Sistema / Paper':<38} {'Recall@5':>10} {'Token F1':>10} {'ROUGE-1':>9}")
    print("  " + "-" * 70)
    print(f"  {'Lewis et al. 2020 — RAG (NQ, EN)':<38} {'80.1 %':>10} {'44.5 %':>10} {'43.2 %':>9}")
    print(f"  {'DPR + FiD (NQ, EN)':<38} {'79.3 %':>10} {'51.4 %':>10} {'—':>9}")
    print(f"  {'RAGAS avg. (multi-domain)':<38} {'75-85 %':>10} {'—':>10} {'—':>9}")
    print(f"  {'Domain RAG típico (literatura)':<38} {'65-90 %':>10} {'35-65 %':>10} {'30-60 %':>9}")
    print("  " + "-" * 70)
    print(f"  {'EducaAssistAI (este benchmark)':<38} {our_r5:>10} {our_f1:>10} {our_r1:>9}")
    print()
    print("  Fontes:")
    print("  · Lewis et al. 2020: 'RAG for Knowledge-Intensive NLP Tasks'. NeurIPS.")
    print("  · Es et al. 2023: 'RAGAS'. arXiv:2309.15217.")
    print("  · Shi et al. 2023: 'Lost in the Middle'. TACL 2023.")
    print()
    print("  Notas metodológicas:")
    print("  · Recall@k: cobertura de docs relevantes nos top-k únicos (IR standard).")
    print("  · Token F1: métrica SQuAD com normalização NFD (PT-PT robusto).")
    print("  · BLEU sentence-level instável para frases curtas; ROUGE-L preferível.")
    print("  · Para list_*: min_results/max_results são as métricas primárias.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://localhost:8000", help="URL base do backend")
    parser.add_argument(
        "--gold",
        default=None,
        help="Path do gold_set_v2.json (default: mesmo directório deste ficheiro)",
    )
    parser.add_argument("--filter", default=None, help="Substring para filtrar IDs (ex: 'fact_')")
    parser.add_argument("--out", default=None, help="Guardar resultados detalhados em JSON")
    parser.add_argument("--timeout", type=int, default=BASE_TIMEOUT, help="Timeout por query (s)")
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        metavar="K",
        help="Valores de k para Recall@k / Precision@k (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=0,
        help="Nº de chunks pedidos ao /retrieve (default: max(k)*3, mín 30). "
             "Separado dos k de avaliação para cobrir corpora grandes.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Só métricas de retrieval (sem chamar /query; muito mais rápido)",
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Só métricas de generation (assume R@k=1.0 para todos os testes)",
    )
    args = parser.parse_args()

    gold_path = args.gold or str(Path(__file__).resolve().parent / "gold_set_v2.json")
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)
    except FileNotFoundError:
        print(f"✗ Gold set não encontrado: {gold_path}", file=sys.stderr)
        print("  Corre: python -m backend.eval.benchmark --gold backend/eval/gold_set_v2.json",
              file=sys.stderr)
        return 2

    if not health_check(args.url):
        print(f"✗ Backend não responde em {args.url}", file=sys.stderr)
        print("  Arranca o backend: uvicorn backend.app:app --reload", file=sys.stderr)
        return 2

    questions: List[Dict] = gold.get("questions", [])
    if args.filter:
        questions = [q for q in questions if args.filter in q["id"]]

    max_k = max(args.k)
    # retrieve_k: chunks pedidos ao /retrieve. Maior que max_k para cobrir
    # duplicados por página (o mesmo URL aparece várias vezes) e garantir
    # que a deduplicação produz pelo menos max_k fontes únicas.
    retrieve_k = args.retrieve_k if args.retrieve_k > 0 else max(max_k * 3, 30)

    print()
    print("═" * 84)
    print("  BENCHMARK EducaAssistAI — AVALIAÇÃO QUANTITATIVA RAG")
    print("═" * 84)
    print(f"  Backend     : {args.url}")
    print(f"  Gold set    : {gold_path}  (v{gold.get('version', '?')})")
    print(f"  Testes      : {len(questions)} / {len(gold.get('questions', []))} total")
    print(f"  k para R@k  : {args.k}  (retrieve_k={retrieve_k})")
    print(f"  Modo        : {'retrieval-only' if args.retrieval_only else 'generation-only' if args.generation_only else 'completo'}")
    print()

    results: List[Dict] = []
    t_start = time.time()

    for i, test in enumerate(questions, start=1):
        print(f"[{i:>2}/{len(questions)}]", end=" ", flush=True)

        retr_scores: Optional[Dict] = None
        gen_scores: Optional[Dict] = None
        notes: List[str] = []
        elapsed_query = 0.0

        # ── Retrieval step ──
        if not args.generation_only:
            retr_resp = retrieve(args.url, test["question"], k=retrieve_k, timeout=args.timeout)
            retr_scores = score_retrieval(test, retr_resp, args.k)

        # ── Generation step ──
        if not args.retrieval_only:
            gen_resp = query(args.url, test["question"], timeout=args.timeout)
            elapsed_query = gen_resp.get("_elapsed_s", 0.0)
            gen_scores, notes = score_generation(test, gen_resp)

        print(f"[{test['id']}]", test["question"][:50])

        print_test_row(test, retr_scores, gen_scores, notes, elapsed_query)

        results.append({
            "test": test,
            "retrieval": retr_scores or {},
            "generation": gen_scores or {},
            "elapsed_query_s": elapsed_query,
            "notes": notes,
        })

    elapsed_total = time.time() - t_start
    print(f"\n  Tempo total: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    print_summary(results, args.k, args.retrieval_only)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def avg(key: str, sub: str) -> float:
            vals = [r[sub].get(key, 0.0) for r in results if r.get(sub)]
            return sum(vals) / len(vals) if vals else 0.0

        summary = {
            "n": len(results),
            "elapsed_total_s": elapsed_total,
            "retrieval": {
                f"recall@{k}": avg(f"recall@{k}", "retrieval") for k in args.k
            } | {
                f"precision@{k}": avg(f"precision@{k}", "retrieval") for k in args.k if k <= 5
            } | {
                "mrr": avg("mrr", "retrieval"),
                "ndcg@5": avg("ndcg@5", "retrieval"),
                "ndcg@10": avg("ndcg@10", "retrieval"),
            },
            "generation": {
                m: avg(m, "generation")
                for m in (
                    "token_f1", "bleu1", "bleu2", "bleu4",
                    "rouge1", "rouge2", "rouge_l", "exact_match",
                    "source_recall", "source_precision",
                )
            },
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        print(f"  Resultados detalhados guardados em: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
