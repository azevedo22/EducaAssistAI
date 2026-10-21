"""
Runner de avaliação do EducaAssistAI.

Faz queries ao backend (que tem de estar em execução em localhost:8000)
para cada pergunta do gold_set.json, mede 3 dimensões por pergunta
(correção, precisão de fontes, aderência a regras), e produz um
relatório agregado por categoria.

USO:
  # Backend a correr em :8000
  python -m backend.eval.run

  # Override do URL ou ficheiro de perguntas
  python -m backend.eval.run --url http://localhost:8000 \\
                              --gold backend/eval/gold_set.json

  # Correr só algumas perguntas (útil para debugging)
  python -m backend.eval.run --filter restr_

  # Guardar resultados detalhados em JSON
  python -m backend.eval.run --out eval_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


# ----------------------------------------------------------------------------
# Cliente HTTP
# ----------------------------------------------------------------------------

def query_backend(url: str, question: str, timeout: int = 300) -> Dict:
    """Faz POST /query e devolve o resultado normalizado."""
    start = time.time()
    try:
        r = requests.post(
            f"{url.rstrip('/')}/query",
            json={"question": question},
            timeout=timeout,
        )
        elapsed = time.time() - start
        r.raise_for_status()
        data = r.json()
        return {
            "ok": True,
            "answer": data.get("answer", ""),
            "sources": data.get("sources", []),
            "elapsed_s": elapsed,
            "status": r.status_code,
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error": f"timeout após {timeout}s",
            "elapsed_s": time.time() - start,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "elapsed_s": time.time() - start,
        }


# ----------------------------------------------------------------------------
# Scorer — três dimensões independentes, valor [0, 1] cada
# ----------------------------------------------------------------------------

def _count_listing_items(answer: str) -> int:
    """Conta itens em formato lista (linhas a começar por '-' ou '•')."""
    n = 0
    for line in answer.split("\n"):
        s = line.strip()
        if s.startswith("- ") or s.startswith("* ") or s.startswith("• "):
            n += 1
    return n


def score_test(test: Dict, response: Dict) -> Dict:
    """Avalia uma resposta segundo o gold entry da pergunta."""
    notes: List[str] = []

    if not response["ok"]:
        return {
            "correctness": 0.0,
            "source_precision": 0.0,
            "rule_adherence": 0.0,
            "notes": [f"backend error: {response.get('error', '?')}"],
        }

    answer = response["answer"]
    answer_low = answer.lower()
    sources = set(response["sources"])

    # ---------- Correção ----------
    correctness_components: List[float] = []

    must_contain = test.get("must_contain", [])
    if must_contain:
        hits = sum(1 for kw in must_contain if kw.lower() in answer_low)
        score_kw = hits / len(must_contain)
        correctness_components.append(score_kw)
        if score_kw < 1.0:
            missed = [kw for kw in must_contain if kw.lower() not in answer_low]
            notes.append(f"missing keywords: {missed}")

    expected_phrase = test.get("expected_answer_phrase")
    if expected_phrase:
        if expected_phrase.lower() in answer_low:
            correctness_components.append(1.0)
        else:
            correctness_components.append(0.0)
            notes.append(f"expected phrase missing: {expected_phrase!r}")

    min_results = test.get("min_results")
    if min_results is not None:
        n_items = _count_listing_items(answer)
        if n_items >= min_results:
            correctness_components.append(1.0)
        else:
            correctness_components.append(min(1.0, n_items / max(1, min_results)))
            notes.append(f"esperava ≥{min_results} itens, vieram {n_items}")

    max_results = test.get("max_results")
    if max_results is not None:
        n_items = _count_listing_items(answer)
        if n_items <= max_results:
            correctness_components.append(1.0)
        else:
            correctness_components.append(0.0)
            notes.append(f"esperava ≤{max_results} itens, vieram {n_items}")

    if not correctness_components:
        # sem critério explícito — assumimos 1.0 (categoria edge ou semantic
        # sem must_contain). O olho humano pode reavaliar.
        correctness = 1.0
    else:
        correctness = sum(correctness_components) / len(correctness_components)

    # ---------- Precisão de fontes ----------
    expected_sources = set(test.get("expected_sources", []))
    if test.get("category") in ("unknown", "anti_hallucination"):
        # Para 'não sei', esperamos NENHUMA fonte (anti-alucinação).
        if not sources:
            source_precision = 1.0
        else:
            source_precision = 0.0
            notes.append(f"fontes inesperadas para {test['category']}: {sources}")
    elif expected_sources:
        intersection = expected_sources & sources
        if expected_sources.issubset(sources):
            source_precision = 1.0
        else:
            source_precision = len(intersection) / len(expected_sources)
            missing = expected_sources - sources
            notes.append(f"fontes em falta: {missing}")
    else:
        source_precision = 1.0

    # ---------- Aderência a regras ----------
    rule_components: List[float] = []

    must_not_contain = test.get("must_not_contain", [])
    if must_not_contain:
        violations = [kw for kw in must_not_contain if kw.lower() in answer_low]
        if not violations:
            rule_components.append(1.0)
        else:
            rule_components.append(0.0)
            notes.append(f"contém termos proibidos: {violations}")

    # Para 'unknown'/'anti_hallucination', a regra é: usar a frase fixa.
    # Já está coberto via expected_answer_phrase em correctness, mas
    # acrescentamos um sinal forte aqui também.
    if test.get("category") in ("unknown", "anti_hallucination"):
        if "não encontrei" in answer_low or len(sources) == 0:
            rule_components.append(1.0)
        else:
            rule_components.append(0.0)
            notes.append("não recusou apesar de não ter informação")

    rule_adherence = (
        sum(rule_components) / len(rule_components)
        if rule_components
        else 1.0
    )

    return {
        "correctness": correctness,
        "source_precision": source_precision,
        "rule_adherence": rule_adherence,
        "notes": notes,
    }


# ----------------------------------------------------------------------------
# Relatório
# ----------------------------------------------------------------------------

def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def print_test_line(test: Dict, response: Dict, scores: Dict) -> None:
    c = scores["correctness"]
    s = scores["source_precision"]
    r = scores["rule_adherence"]
    overall = (c + s + r) / 3
    elapsed = response.get("elapsed_s", 0)

    status = "✓" if overall >= 0.8 else ("~" if overall >= 0.5 else "✗")
    qid = test["id"]
    qtxt = test["question"]
    if len(qtxt) > 70:
        qtxt = qtxt[:67] + "..."

    print(f"  {status} [{qid:<10}] {qtxt}")
    print(
        f"     correção={c:.2f} {_bar(c, 12)}  "
        f"fontes={s:.2f} {_bar(s, 12)}  "
        f"regras={r:.2f} {_bar(r, 12)}  "
        f"({elapsed:.1f}s)"
    )
    for note in scores["notes"][:2]:
        print(f"     · {note}")


def print_summary(results: List[Dict]) -> None:
    n = len(results)
    if n == 0:
        print("Nenhum teste executado.")
        return

    avg_c = sum(r["scores"]["correctness"] for r in results) / n
    avg_s = sum(r["scores"]["source_precision"] for r in results) / n
    avg_r = sum(r["scores"]["rule_adherence"] for r in results) / n
    avg_t = sum(r["response"].get("elapsed_s", 0) for r in results) / n
    overall = (avg_c + avg_s + avg_r) / 3

    print()
    print("═" * 80)
    print("  RESUMO GLOBAL")
    print("═" * 80)
    print(f"  Total de testes:        {n}")
    print(f"  Tempo médio por query:  {avg_t:.1f}s")
    print()
    print(f"  Correção:               {avg_c:.2%}  {_bar(avg_c)}")
    print(f"  Precisão de fontes:     {avg_s:.2%}  {_bar(avg_s)}")
    print(f"  Aderência a regras:     {avg_r:.2%}  {_bar(avg_r)}")
    print(f"  ─" * 39)
    print(f"  Overall:                {overall:.2%}  {_bar(overall)}")
    print()

    # Por categoria
    cats: Dict[str, List] = {}
    for r in results:
        cat = r["test"].get("category", "uncat")
        cats.setdefault(cat, []).append(r)

    print("  Por categoria:")
    print(f"  {'categoria':<22} {'n':>3} {'correção':>11} {'fontes':>11} {'regras':>11}")
    for cat in sorted(cats):
        rs = cats[cat]
        c = sum(x["scores"]["correctness"] for x in rs) / len(rs)
        s = sum(x["scores"]["source_precision"] for x in rs) / len(rs)
        rr = sum(x["scores"]["rule_adherence"] for x in rs) / len(rs)
        print(f"  {cat:<22} {len(rs):>3} {c:>10.2%}  {s:>10.2%}  {rr:>10.2%}")
    print()

    # Falhas notáveis
    fails = [
        r for r in results
        if (r["scores"]["correctness"] + r["scores"]["source_precision"] +
            r["scores"]["rule_adherence"]) / 3 < 0.6
    ]
    if fails:
        print(f"  Testes com pontuação <60% ({len(fails)} testes):")
        for r in fails:
            print(f"    · [{r['test']['id']}] {r['test']['question'][:60]}")
        print()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000",
                        help="URL base do backend (default: http://localhost:8000)")
    parser.add_argument("--gold", default=None,
                        help="Path para o gold_set.json (default: backend/eval/gold_set.json)")
    parser.add_argument("--filter", default=None,
                        help="Substring para filtrar IDs (ex: 'restr_' corre só restritivos)")
    parser.add_argument("--out", default=None,
                        help="Guardar resultados detalhados em JSON")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout por query, em segundos")
    args = parser.parse_args()

    gold_path = args.gold or str(
        Path(__file__).resolve().parent / "gold_set.json"
    )
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)

    # Health-check antes de começar
    try:
        h = requests.get(f"{args.url.rstrip('/')}/health", timeout=5)
        h.raise_for_status()
    except Exception as exc:
        print(f"✗ Backend não responde em {args.url}: {exc}", file=sys.stderr)
        print("  Arranca o uvicorn antes de correr a avaliação.", file=sys.stderr)
        return 2

    questions = gold["questions"]
    if args.filter:
        questions = [q for q in questions if args.filter in q["id"]]

    print()
    print("═" * 80)
    print("  AVALIAÇÃO DO EducaAssistAI")
    print("═" * 80)
    print(f"  Backend:  {args.url}")
    print(f"  Gold set: {gold_path}")
    print(f"  Testes:   {len(questions)} (de {len(gold['questions'])} totais)")
    print(f"  Timeout:  {args.timeout}s por query")
    print()

    results: List[Dict] = []
    t_start = time.time()

    for i, test in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}]")
        response = query_backend(args.url, test["question"], timeout=args.timeout)
        scores = score_test(test, response)
        results.append({"test": test, "response": response, "scores": scores})
        print_test_line(test, response, scores)
        print()

    elapsed_total = time.time() - t_start
    print(f"  Tempo total: {elapsed_total:.0f}s ({elapsed_total / 60:.1f}min)")

    print_summary(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "n": len(results),
                        "correctness": sum(r["scores"]["correctness"] for r in results) / max(1, len(results)),
                        "source_precision": sum(r["scores"]["source_precision"] for r in results) / max(1, len(results)),
                        "rule_adherence": sum(r["scores"]["rule_adherence"] for r in results) / max(1, len(results)),
                        "elapsed_avg_s": sum(r["response"].get("elapsed_s", 0) for r in results) / max(1, len(results)),
                        "elapsed_total_s": elapsed_total,
                    },
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        print(f"  Resultados detalhados em: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
