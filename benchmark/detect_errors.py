#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detecta linhas com respostas contaminadas (erros/vazias) em resultados.json."""
import json, re, sys

ERROR_PATTERNS = [
    r"\[modo extrativo",
    r"erro do groq",
    r"rate limit",
    r"erro de servi",
    r"429",
    r"413",
    r"not found",
    r"não encontrei essa informação",   # recusa indevida em categorias que deviam responder
]

def is_bad(row):
    resp = (row.get("resposta") or "").strip()
    if not resp:
        return "vazia"
    rl = resp.lower()
    for pat in ERROR_PATTERNS:
        if re.search(pat, rl):
            return f"padrão: {pat}"
    return None

path = sys.argv[1] if len(sys.argv) > 1 else "resultados.json"
data = json.load(open(path, encoding="utf-8"))["linhas"]

bad = []
for r in data:
    reason = is_bad(r)
    if reason:
        bad.append((r["id"], r["categoria"], reason, (r.get("resposta") or "")[:80]))

print(f"Total contaminadas: {len(bad)}")
print()
for rid, cat, reason, preview in sorted(bad):
    print(f"  {rid:20s} [{cat:15s}]  {reason}")
    print(f"    → {preview!r}")

print()
print("IDs para regenerar:")
print(",".join(rid for rid, *_ in bad))
