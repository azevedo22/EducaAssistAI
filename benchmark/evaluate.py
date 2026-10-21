#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - Avaliacao do EducaAssistAI sobre o gold standard (Capitulo 4).

Tres regimes de metrica, conforme a categoria:
  - factual / listagem / restricao / multi_hop / conversacional
        -> recuperacao (Recall@k, NDCG@k, MRR via evidencia) + geracao (Token F1, ROUGE-L, BLEU)
  - desconhecido / anti_alucinacao
        -> taxa de recusa correta (o sistema absteve-se?)
  - contagem
        -> acerto numerico exato (e +/-1, reportado a parte)

Uso:
  python evaluate.py --gold gold_standard_corrigido.xlsx
  python evaluate.py --gold-json ../backend/eval/gold_set_v2.json
  python evaluate.py --gold-json ../backend/eval/gold_set_v2.json --limit 2
  python evaluate.py --selftest
"""

import argparse, json, math, re, sys, time, unicodedata
from collections import defaultdict

# Intervalo entre perguntas (segundos).
# Groq free tier llama-3.1-8b-instant: 6 000 TPM.  Com ~3 500 tokens/pedido
# e 2 chamadas por query (translate + synthesis), precisamos de ~42s de janela
# deslizante para recuperar. 15s é conservador mas evita 429/413.
INTER_QUERY_SLEEP = 60

# =============================================================================
# 1) ADAPTER
# =============================================================================
# /query  -> POST {"question": str}  -> {"answer": str, "sources": List[str]}
# /retrieve -> POST {"question": str, "k": int} -> {"chunks": [{"text","source"}]}

QUERY_ENDPOINT         = "http://127.0.0.1:8000/query"
RETRIEVE_ENDPOINT      = "http://127.0.0.1:8000/retrieve"
CONVERSATIONS_ENDPOINT = "http://127.0.0.1:8000/conversations"

def query_system(pergunta, historico=None):
    import requests

    conversation_id = None
    retrieve_question = pergunta

    # Para perguntas conversacionais: criar uma conversa, enviar os turnos
    # anteriores para construir o histórico, depois enviar a pergunta real.
    # Sem isto, "E em que departamento está?" chega ao backend sem contexto.
    if historico:
        r0 = requests.post(CONVERSATIONS_ENDPOINT, json={"title": "eval-tmp"}, timeout=10)
        r0.raise_for_status()
        conversation_id = r0.json().get("id")

        for turn in historico:
            if turn.get("role") == "user":
                requests.post(QUERY_ENDPOINT, json={
                    "question": turn["content"],
                    "conversation_id": conversation_id,
                }, timeout=600)
                time.sleep(2)

        # /retrieve não suporta conversation_id — enviar a pergunta composta
        # para que o retriever tenha o contexto necessário.
        prior = " ".join(t["content"] for t in historico if t.get("role") == "user")
        retrieve_question = prior + " " + pergunta

    # 1) Resposta LLM via /query
    payload = {"question": pergunta}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    r = requests.post(QUERY_ENDPOINT, json=payload, timeout=600)
    r.raise_for_status()
    resposta = r.json().get("answer") or ""

    # 2) Chunks rankeados via /retrieve (k=20 para cobrir recall@20)
    r2 = requests.post(RETRIEVE_ENDPOINT, json={"question": retrieve_question, "k": 20}, timeout=30)
    r2.raise_for_status()
    chunks = [
        {"texto": c.get("text", ""), "fonte": c.get("source", "")}
        for c in (r2.json().get("chunks") or [])
    ]
    return {"resposta": resposta, "chunks": chunks}

# =============================================================================
# 2) NORMALIZACAO pt-PT
# =============================================================================
def norm(s):
    if s is None: return ""
    s = str(s).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s/:-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s):
    return norm(s).split()

# =============================================================================
# 3) METRICAS DE RECUPERACAO
# =============================================================================
def split_evidence(ev):
    return [e.strip() for e in str(ev or "").split("|") if e.strip()]

def chunk_has(chunk_text_norm, evidence_item):
    return norm(evidence_item) in chunk_text_norm

def relevance_flags(chunks, evidence_items, k):
    flags = []
    for c in chunks[:k]:
        ct = norm(c.get("texto", ""))
        flags.append(1 if any(chunk_has(ct, it) for it in evidence_items) else 0)
    return flags

def recall_at_k(chunks, evidence_items, k):
    if not evidence_items: return None
    topk = [norm(c.get("texto", "")) for c in chunks[:k]]
    found = sum(1 for it in evidence_items if any(chunk_has(ct, it) for ct in topk))
    return found / len(evidence_items)

def ndcg_at_k(chunks, evidence_items, k):
    if not evidence_items: return None
    rel = relevance_flags(chunks, evidence_items, k)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    # IDCG: ideal onde todos os chunks relevantes ficam no topo.
    # n_rel = chunks relevantes encontrados no top-k (lower bound do corpus).
    # Usar len(evidence_items) causaria NDCG > 1 quando um único item de
    # evidência aparece em múltiplos chunks.
    n_rel = sum(rel)
    if n_rel == 0:
        return 0.0
    idcg = sum(1 / math.log2(i + 2) for i in range(n_rel))
    return (dcg / idcg) if idcg > 0 else 0.0

def mrr(chunks, evidence_items):
    if not evidence_items: return None
    for i, c in enumerate(chunks):
        if any(chunk_has(norm(c.get("texto", "")), it) for it in evidence_items):
            return 1 / (i + 1)
    return 0.0

def both_docs_retrieved(chunks, docs):
    fontes = {norm(c.get("fonte", "")) for c in chunks}
    needed = [norm(d) for d in re.split(r"[|]", str(docs)) if d.strip()]
    if not needed:
        return None
    def _matches(nd):
        # Correspondência exacta por substring
        if any(nd in f or f in nd for f in fontes):
            return True
        # Correspondência por prefixo do primeiro token:
        # 'horarios pdf' -> 'horarios' está contido em
        # 'horarios_de_atendimento_1 _sem_ 2025_2026 pdf'
        prefix = nd.split()[0] if nd.split() else ""
        return bool(prefix) and any(prefix in f for f in fontes)
    return all(_matches(nd) for nd in needed)

# =============================================================================
# 4) METRICAS DE GERACAO
# =============================================================================
def token_f1(hyp, ref):
    h, r = tokens(hyp), tokens(ref)
    if not h or not r: return 0.0
    common = defaultdict(int)
    hc = defaultdict(int)
    for t in h: hc[t] += 1
    rc = defaultdict(int)
    for t in r: rc[t] += 1
    overlap = sum(min(hc[t], rc[t]) for t in hc)
    if overlap == 0: return 0.0
    prec, rec = overlap / len(h), overlap / len(r)
    return 2 * prec * rec / (prec + rec)

def rouge_l(hyp, ref):
    h, r = tokens(hyp), tokens(ref)
    if not h or not r: return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(1, len(h) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if h[i-1] == r[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[len(h)][len(r)]
    if lcs == 0: return 0.0
    prec, rec = lcs / len(h), lcs / len(r)
    return 2 * prec * rec / (prec + rec)

def bleu(hyp, ref, n_max=4):
    h, r = tokens(hyp), tokens(ref)
    if not h: return 0.0
    precs = []
    for n in range(1, n_max + 1):
        hg = [tuple(h[i:i+n]) for i in range(len(h)-n+1)]
        rg = [tuple(r[i:i+n]) for i in range(len(r)-n+1)]
        if not hg:
            precs.append(0.0); continue
        rc = defaultdict(int)
        for g in rg: rc[g] += 1
        match = 0
        used = defaultdict(int)
        for g in hg:
            if used[g] < rc[g]:
                match += 1; used[g] += 1
        precs.append((match + 1) / (len(hg) + 1))
    # Resposta mais curta que n_max tokens produz prec=0.0 → log(0) → crash.
    if any(p <= 0 for p in precs):
        return 0.0
    geo = math.exp(sum(math.log(p) for p in precs) / n_max)
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r)/max(len(h), 1))
    return bp * geo

# =============================================================================
# 5) RECUSA / CONTAGEM
# =============================================================================
REFUSAL_MARKERS = [norm(m) for m in [
    "nao consta", "nao encontrei", "nao tenho informacao", "nao disponho",
    "nao foi possivel", "nao consigo", "sem informacao", "nao esta disponivel",
    "nao posso responder", "nao existe", "nao ha informacao", "nao tenho dados",
    "nao foi encontrada", "desconheco", "nao e possivel determinar",
]]
CORRECTION_MARKERS = [norm(m) for m in [
    "incorreto", "incorreta", "falso", "nao e verdade", "na verdade",
    "corrigir", "esta errado", "nao confirmo", "nao corresponde",
]]

def refused(resposta, allow_correction=False):
    a = norm(resposta)
    if not a: return True
    markers = REFUSAL_MARKERS + (CORRECTION_MARKERS if allow_correction else [])
    return any(m in a for m in markers)

def extract_int(resposta):
    m = re.search(r"\d+", str(resposta).replace(".", "").replace(",", ""))
    return int(m.group()) if m else None

# =============================================================================
# 6) AVALIACAO POR LINHA
# =============================================================================
RETRIEVAL_CATS = {"factual", "listagem", "restricao", "multi_hop", "conversacional"}
REFUSAL_CATS   = {"desconhecido", "anti_alucinacao"}

def evaluate_row(row, ks=(5, 20)):
    cat = row["categoria"]
    hist = None
    if cat == "conversacional" and row.get("contexto_anterior"):
        hist = [{"role": "user", "content": row["contexto_anterior"]}]
    out = query_system(row["pergunta"], historico=hist)
    resp, chunks = out["resposta"], out["chunks"]
    res = {"id": row["id"], "categoria": cat, "resposta": resp}

    if cat in RETRIEVAL_CATS:
        ev = split_evidence(row["evidencia"])
        for k in ks:
            res[f"recall@{k}"] = recall_at_k(chunks, ev, k)
        res["ndcg@10"] = ndcg_at_k(chunks, ev, 10)
        res["mrr"] = mrr(chunks, ev)
        res["token_f1"] = token_f1(resp, row["resposta_esperada"])
        res["rouge_l"] = rouge_l(resp, row["resposta_esperada"])
        res["bleu"] = bleu(resp, row["resposta_esperada"])
        if cat == "multi_hop":
            res["ambos_docs"] = both_docs_retrieved(chunks, row.get("documento", ""))
        if cat == "conversacional" and "MUDANCA DE TOPICO" in norm(row.get("notas", "")).upper().replace("Ç","C"):
            res["topic_change"] = True
    elif cat in REFUSAL_CATS:
        res["recusou"] = refused(resp, allow_correction=(cat == "anti_alucinacao"))
    elif cat == "contagem":
        got, exp = extract_int(resp), extract_int(row["resposta_esperada"])
        res["num_sistema"], res["num_ref"] = got, exp
        res["exato"] = (got is not None and got == exp)
        res["tol1"] = (got is not None and exp is not None and abs(got - exp) <= 1)
    return res

# =============================================================================
# 7) AGREGACAO + LATEX
# =============================================================================
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

def aggregate(results):
    by = defaultdict(list)
    for r in results: by[r["categoria"]].append(r)
    agg = {}
    for cat, rs in by.items():
        if cat in RETRIEVAL_CATS:
            agg[cat] = {
                "n": len(rs),
                "recall@5": mean([r.get("recall@5") for r in rs]),
                "recall@20": mean([r.get("recall@20") for r in rs]),
                "ndcg@10": mean([r.get("ndcg@10") for r in rs]),
                "mrr": mean([r.get("mrr") for r in rs]),
                "token_f1": mean([r.get("token_f1") for r in rs]),
                "rouge_l": mean([r.get("rouge_l") for r in rs]),
                "bleu": mean([r.get("bleu") for r in rs]),
            }
        elif cat in REFUSAL_CATS:
            agg[cat] = {"n": len(rs), "taxa_recusa": mean([1.0 if r["recusou"] else 0.0 for r in rs])}
        elif cat == "contagem":
            agg[cat] = {"n": len(rs),
                        "acerto_exato": mean([1.0 if r["exato"] else 0.0 for r in rs]),
                        "acerto_tol1": mean([1.0 if r["tol1"] else 0.0 for r in rs])}
    return agg

def fmt(x): return "---" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2f}"

def to_latex(agg):
    L = []
    order = ["factual", "listagem", "restricao", "multi_hop", "conversacional"]
    L.append("% --- Recuperacao + Geracao ---")
    L.append("\\begin{table}[htbp]\\centering\\renewcommand{\\arraystretch}{1.3}")
    L.append("\\rowcolors{2}{ipbrow}{white}")
    L.append("\\begin{tabular}{l c c c c c c}")
    L.append("\\rowcolor{ipbvinho}")
    L.append("\\textcolor{white}{Categoria} & \\textcolor{white}{R@5} & \\textcolor{white}{R@20} & "
             "\\textcolor{white}{NDCG@10} & \\textcolor{white}{MRR} & \\textcolor{white}{Token F1} & \\textcolor{white}{ROUGE-L}\\\\")
    for c in order:
        if c in agg:
            a = agg[c]
            L.append(f"{c} & {fmt(a['recall@5'])} & {fmt(a['recall@20'])} & {fmt(a['ndcg@10'])} & "
                     f"{fmt(a['mrr'])} & {fmt(a['token_f1'])} & {fmt(a['rouge_l'])}\\\\")
    L.append("\\bottomrule\\end{tabular}")
    L.append("\\caption{Resultados de recuperacao e geracao por categoria.}\\label{tab:resultados-principais}")
    L.append("\\end{table}")
    L.append("")
    L.append("% --- Recusa ---")
    for c in ["desconhecido", "anti_alucinacao"]:
        if c in agg:
            L.append(f"% {c}: taxa de recusa correta = {fmt(agg[c]['taxa_recusa'])} (n={agg[c]['n']})")
    if "contagem" in agg:
        a = agg["contagem"]
        L.append(f"% contagem: acerto exato = {fmt(a['acerto_exato'])}, +/-1 = {fmt(a['acerto_tol1'])} (n={a['n']})")
    return "\n".join(L)

# =============================================================================
# 8) LOADERS
# =============================================================================

# Mapeamento de categorias do JSON para as categorias internas do evaluate.py
_CAT_MAP = {
    "factual":           "factual",
    "listing":           "listagem",
    "restrictive":       "restricao",
    "semantic":          "restricao",   # refinamento semântico trata-se como restricao
    "unknown":           "desconhecido",
    "anti_hallucination":"anti_alucinacao",
    "count":             "contagem",
    "conversational":    "conversacional",
    "multi_hop":         "multi_hop",
}

def load_gold_json(path):
    """Carrega gold set do formato JSON (gold_set_v2.json) e converte
    para o formato de linha esperado por evaluate_row."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for q in data.get("questions", []):
        cat_raw = q.get("category", "factual")
        cat = _CAT_MAP.get(cat_raw, cat_raw)

        must = q.get("must_contain") or []
        evidencia = "|".join(str(m) for m in must) if must else ""

        sources = q.get("expected_sources") or []
        documento = "|".join(sources)

        rows.append({
            "id":               q.get("id", ""),
            "categoria":        cat,
            "pergunta":         q.get("question", ""),
            "resposta_esperada":q.get("reference_answer", ""),
            "evidencia":        evidencia,
            "documento":        documento,
            "contexto_anterior":q.get("contexto_anterior") or None,
            "notas":            q.get("notes", "") or "",
        })
    return rows


def load_gold_xlsx(path):
    """Carrega gold set do formato .xlsx (folha 'Perguntas')."""
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True)
    ws = wb["Perguntas"] if "Perguntas" in wb.sheetnames else wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    H = [str(h) for h in rows[0]]
    out = []
    for r in rows[1:]:
        if not any(r): continue
        out.append({H[i]: r[i] for i in range(len(H))})
    return out


# =============================================================================
# 9) RUNNER + MAIN
# =============================================================================
def run(data, limit=None, ids=None, merge=False, ks=(5, 20)):
    if limit:
        seen = defaultdict(int); kept = []
        for d in data:
            if seen[d["categoria"]] < limit:
                kept.append(d); seen[d["categoria"]] += 1
        data = kept

    if ids:
        id_set = set(ids)
        data = [d for d in data if d["id"] in id_set]

    results = []
    for i, row in enumerate(data, 1):
        print(f"[{i}/{len(data)}] {row['id']} ({row['categoria']}) ...", file=sys.stderr)
        try:
            results.append(evaluate_row(row, ks=ks))
        except Exception as e:
            print(f"  ERRO em {row['id']}: {e}", file=sys.stderr)
        if i < len(data):
            time.sleep(INTER_QUERY_SLEEP)

    # Modo merge: carrega resultados anteriores e substitui/acrescenta linhas novas.
    if merge:
        out_path = "resultados.json"
        try:
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f)
            prev_lines = prev.get("linhas", [])
            prev_ids = {r["id"] for r in results}
            merged = [r for r in prev_lines if r["id"] not in prev_ids] + results
            results = merged
        except FileNotFoundError:
            pass

    agg = aggregate(results)
    print("\n==== RESUMO ====")
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print("\n==== LATEX ====")
    print(to_latex(agg))

    out_path = "resultados.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"linhas": results, "agregado": agg}, f, ensure_ascii=False, indent=2)
    print(f"\n(guardado em {out_path})", file=sys.stderr)


def selftest():
    global query_system
    def mock(pergunta, historico=None):
        return {"resposta": "Resposta de teste 15/09/2025 Joao",
                "chunks": [{"texto": "Joao Silva atende 15/09/2025", "fonte": "horarios.pdf"},
                           {"texto": "outro chunk", "fonte": "CalendarioAcademico.pdf"}]}
    query_system = mock
    rows = [
        {"id":"f1","categoria":"factual","pergunta":"x","resposta_esperada":"Joao atende 15/09/2025",
         "evidencia":"Joao|15/09/2025","documento":"horarios.pdf","contexto_anterior":None,"notas":""},
        {"id":"u1","categoria":"desconhecido","pergunta":"x","resposta_esperada":"deve recusar",
         "evidencia":"","documento":"","contexto_anterior":None,"notas":""},
        {"id":"c1","categoria":"contagem","pergunta":"quantos","resposta_esperada":15,
         "evidencia":"","documento":"horarios.pdf","contexto_anterior":None,"notas":""},
    ]
    res = [evaluate_row(r) for r in rows]
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nSELF-TEST OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold",      default=None, help="Ficheiro .xlsx com o gold standard")
    ap.add_argument("--gold-json", default=None, help="Ficheiro .json com o gold standard (gold_set_v2.json)")
    ap.add_argument("--limit",     type=int, default=None, help="N perguntas por categoria (teste rapido)")
    ap.add_argument("--ids",       nargs="+", default=None, help="Re-executar apenas estes IDs (ex: fact_15 fact_16)")
    ap.add_argument("--merge",     action="store_true", help="Fundir novos resultados com resultados.json existente")
    ap.add_argument("--selftest",  action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
    elif args.gold_json:
        run(load_gold_json(args.gold_json), limit=args.limit, ids=args.ids, merge=args.merge)
    elif args.gold:
        run(load_gold_xlsx(args.gold), limit=args.limit, ids=args.ids, merge=args.merge)
    else:
        ap.error("Especifica --gold ou --gold-json (ou --selftest para validar a logica)")
