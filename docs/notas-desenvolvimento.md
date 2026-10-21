# EducaAssistAI

Assistente inteligente para consulta de documentação institucional do IPB.

## Instalação

1. Criar ambiente virtual:
   ```bash
   python -m venv .venv
   ```

2. Ativar ambiente:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

3. Instalar dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Configuração

1. Preparar dados:
   - Adicionar PDFs em `backend/data/`
   - Ou usar o arquivo de exemplo `sample.txt`

2. Construir índice:
   ```bash
   cd backend
   python prepare_index.py
   ```

## Execução

1. Backend:
   ```bash
   uvicorn backend.app:app --reload
   ```

2. Frontend:
   ```bash
   cd frontend
   npm install
   npm start
   ```

## Fornecedor LLM (Ollama local vs Groq cloud)

O sistema suporta dois fornecedores intercambiáveis para o LLM, controlados
pela variável de ambiente `LLM_PROVIDER`:

- **`ollama`** (default) — modelo local em CPU. Sem custos, sem dependência
  de internet, mas lento (5–30 tok/s). Útil para desenvolvimento offline.
- **`groq`** — Groq Cloud. Hardware LPU dedicado, ~200–700 tok/s
  (50–100× mais rápido). Tier gratuito sem cartão de crédito. Ideal para
  demonstrações e benchmark de qualidade.

### Usar Ollama (local) — default

Não é preciso configurar nada. O backend assume `LLM_PROVIDER=ollama` e o
modelo `gemma2:2b`. Garante que o `ollama serve` está a correr.

```bash
ollama pull gemma2:2b
ollama serve            # noutra janela
uvicorn backend.app:app --reload
```

Variáveis opcionais: `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_KEEP_ALIVE`,
`OLLAMA_TIMEOUT`.

### Usar Groq (cloud, gratuito)

1. Cria conta em https://console.groq.com (sem cartão de crédito).
2. Em https://console.groq.com/keys gera uma chave API.
3. Define as env vars antes de arrancar o backend:

   ```bash
   # PowerShell
   $env:LLM_PROVIDER = "groq"
   $env:GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
   uvicorn backend.app:app --reload

   # cmd
   set LLM_PROVIDER=groq
   set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   uvicorn backend.app:app --reload

   # bash / WSL
   export LLM_PROVIDER=groq
   export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   uvicorn backend.app:app --reload
   ```

Variáveis opcionais: `GROQ_MODEL` (default `llama-3.3-70b-versatile`),
`GROQ_URL`, `GROQ_TIMEOUT`.

Modelos recomendados no tier gratuito:

- `llama-3.3-70b-versatile` — melhor qualidade, ainda muito rápido (~150 tok/s).
- `llama-3.1-8b-instant` — alternativa mais rápida (~500 tok/s) com
  ligeira perda de qualidade.

Para confirmar qual fornecedor está em uso, abre o frontend e verifica o
**badge no canto superior direito** (Ollama: fundo translúcido, ponto
verde; Groq: fundo branco, ponto azul). Em alternativa:

```bash
curl http://localhost:8000/health/llm
```

### Comparação Ollama vs Groq (para o relatório)

A mesma pergunta pode ser respondida pelos dois fornecedores sem alterar
código — basta reiniciar o backend com `LLM_PROVIDER` diferente. Sugestões
de métricas a registar:

- Latência fim-a-fim (`POST /query`).
- Latência só do LLM (subtraindo o tempo de retrieval).
- Qualidade subjetiva (1–5) e cobertura factual da resposta.
- Sucesso do tradutor estruturado (`[query] tradutor LLM:` no log).

## Pipeline de extração de PDFs (markdown vs texto vs tabela)

Os PDFs carregados passam por um pipeline de 3 modos com fallback
automático, cada um com vantagens diferentes:

| Modo | Biblioteca | Quando é usado | Ideal para |
|---|---|---|---|
| **`markdown`** (default) | PyMuPDF4LLM | Auto-deteção quando a extração tabular pura produz chunks ilegíveis | PDFs híbridos com prosa + tabelas (calendários, regulamentos, propostas) |
| **`table`** | PyMuPDF `find_tables` | Auto-deteção quando os chunks tabulares têm pelo menos 2 palavras "narrativas" por linha | Horários, listas com identidade clara em cada linha |
| **`text`** | PyMuPDF `get_text` | Fallback se markdown falhar | PDFs simples só com texto |

A heurística automática avalia a qualidade da extração tabular: se a
maioria dos chunks tem só códigos curtos e números (como
`SEG (15, INÍCIO, 13, 10)`), o modo é descartado a favor de markdown.

### Por que PyMuPDF4LLM como default?

Avaliámos quatro alternativas para extração rica:

| Ferramenta | Pros | Contras | Veredicto |
|---|---|---|---|
| **PyMuPDF4LLM** | Mesma equipa do PyMuPDF; markdown unificado; sem deps adicionais | Tabelas grandes podem perder formatação | ✅ Escolhido |
| Unstructured.io | Qualidade muito alta; suporta múltiplos formatos | ~500 MB de dependências (tesseract, libmagic, poppler) | Adiado |
| Camelot | Excelente para tabelas isoladas | Só extrai tabelas; precisa de Ghostscript; ignora prosa | Não cabe |
| Vision LLM | Qualidade máxima, lê layouts complexos | Quota cloud; latência; custo recorrente | Considerar como Phase 2 |

### Re-extrair manualmente

Se a auto-deteção escolher mal para um PDF específico, podes forçar
qualquer modo via UI (botão de download no card do documento) ou via
API:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"mode":"markdown"}' \
     http://localhost:8000/documents/MeuPdf.pdf/reextract
```

Substitui `markdown` por `text` ou `table` para os outros modos.

## Provider de embeddings (sentence-transformers local vs Gemini cloud)

Independentemente do LLM, o sistema também tem dois fornecedores
intercambiáveis para gerar embeddings, controlados por `EMBED_PROVIDER`:

- **`local`** (default) — `paraphrase-multilingual-MiniLM-L12-v2` em CPU
  via sentence-transformers. 384 dimensões, ~471 MB descarregados,
  ~25–60 chunks/segundo. Sem custos, sem internet.
- **`gemini`** — `text-embedding-004` da Google. 768 dimensões. Tier
  gratuito: 1500 req/min, sem cap diário. Necessita `GOOGLE_API_KEY`
  (https://aistudio.google.com/apikey, sem cartão de crédito).

**Aviso técnico — coleções separadas.** Vetores de dimensões e espaços
semânticos diferentes não são compatíveis: o sistema mantém **uma coleção
ChromaDB por provider** (`ipb_docs_v3_local` e `ipb_docs_v3_gemini`).
Trocar de provider implica re-carregar os PDFs nesse modo (uma vez), mas
em troca podes manter os dois índices em paralelo e alternar
instantaneamente para comparar respostas no relatório.

### Usar embeddings locais — default

Não é preciso configurar nada. A primeira chamada descarrega o modelo
para `~/.cache/huggingface/`. Todos os PDFs já carregados continuam
disponíveis nesta coleção.

### Usar embeddings Gemini

```bash
# PowerShell
$env:EMBED_PROVIDER = "gemini"
$env:GOOGLE_API_KEY = "AIza..."
uvicorn backend.app:app --reload

# bash / WSL
export EMBED_PROVIDER=gemini
export GOOGLE_API_KEY=AIza...
uvicorn backend.app:app --reload
```

Variáveis opcionais: `GEMINI_EMBED_MODEL` (default `models/text-embedding-004`).

Diagnóstico via curl:

```bash
curl http://localhost:8000/health/embeddings
```

A UI mostra um **segundo badge** ao lado do badge do LLM (mesmo código de
cores: branco/azul para cloud, translúcido/verde para local) com o
provider, o modelo e a coleção em uso. Bom material para screenshots
do relatório.

### Combinações úteis para o relatório

| LLM | Embeddings | Cenário |
|---|---|---|
| Ollama | Local | Totalmente offline. Lento mas independente. Baseline. |
| Groq | Local | Resposta rápida, embeddings em cache local. Equilibrado. |
| Ollama | Gemini | Embeddings de alta qualidade mas síntese lenta. Pouco útil. |
| Groq | Gemini | Tudo cloud. Mais rápido e qualidade máxima — referência ideal. |

Sugestões de métricas a registar para cada combinação:

- Latência fim-a-fim (`POST /query`).
- Latência só do retrieval vs só do LLM.
- Qualidade subjetiva (1–5) e cobertura factual.
- Recall@k qualitativo: top-3 chunks contêm a resposta certa?

## API

- `GET /health` - Verificar saúde geral
- `GET /health/llm` - Estado do LLM (provider, modelo, disponibilidade)
- `GET /health/embeddings` - Estado dos embeddings (provider, modelo, dim, coleção, contagem)
- `POST /query` - Consultar com pergunta
- `POST /upload` - Carregar PDF
- `GET /documents` - Listar PDFs carregados
- `DELETE /documents/{filename}` - Apagar documento
- `PATCH /documents/{filename}/rename` - Renomear documento
- `GET /documents/{filename}/file` - Servir PDF para visualização

## Estrutura

- `backend/` - API Python com FastAPI
  - `llm_client.py` - Abstração unificada Ollama/Groq
  - `app.py` - Endpoints REST
  - `answer.py` - Composição de respostas via LLM
  - `retrieval.py` / `pdf_tables.py` / `chunk.py` - Pipeline RAG
  - `schema_store.py` - Registo persistente de esquemas tabulares
- `frontend/` - Interface Angular
- `requirements.txt` - Dependências Python
- `frontend/package.json` - Dependências Node.js