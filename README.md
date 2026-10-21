# EducaAssistAI

Assistente conversacional com RAG (Retrieval-Augmented Generation) para consulta de documentação institucional do IPB. Responde com base nos documentos carregados e recusa responder quando a informação não existe nesses documentos.

> [Captura de ecrã ou GIF de uma conversa]

## Funcionalidades

- Perguntas em linguagem natural sobre documentos institucionais (PDF e texto)
- Pesquisa híbrida: vetorial (ChromaDB) e léxica (BM25), com os resultados fundidos por Reciprocal Rank Fusion
- Extração de PDFs em três modos (markdown, tabela, texto), com seleção automática e fallback
- Mecanismo de recusa: quando não há evidência nos documentos, o sistema não inventa uma resposta
- LLM e embeddings intercambiáveis por configuração

## Arquitetura

```
Angular (frontend)
      │  HTTP
      ▼
FastAPI (backend)
      ├── Pipeline RAG: recuperação híbrida → composição da resposta via LLM
      ├── Pipeline de ingestão: extração de PDFs → chunking → embeddings
      ▼
ChromaDB (vetores) · BM25 (léxico) · SQLite (conversas e metadados)
```

Todos os clientes de LLM implementam a interface `BaseLLMClient`, por isso trocar de fornecedor não altera a lógica RAG.

## Requisitos

- Python 3.9 ou superior (desenvolvido com 3.9.13)
- Node.js (LTS compatível com Angular 17)
- [Ollama](https://ollama.com), apenas se usares o LLM local

## Instalação

Backend (na raiz do projeto):

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Frontend:

```bash
cd frontend
npm install
```

## Configuração

Copia `.env.example` para `.env` e preenche o que precisares. O ficheiro `.env` contém chaves de API e não deve ser commitado.

| Variável | Valores | Por defeito | Descrição |
|---|---|---|---|
| `LLM_PROVIDER` | `ollama`, `groq`, `gemini` | `ollama` | Fornecedor do LLM |
| `EMBED_PROVIDER` | `local`, `gemini` | `local` | Fornecedor dos embeddings |
| `OLLAMA_URL` / `OLLAMA_MODEL` | | `http://localhost:11434` / `gemma2:2b` | Ollama local |
| `GROQ_API_KEY` / `GROQ_MODEL` | | `llama-3.3-70b-versatile` | Groq (cloud, tier gratuito) |
| `GEMINI_LLM_API_KEY` / `GEMINI_LLM_MODEL` | | `gemini-2.5-flash` | Gemini como LLM |
| `GOOGLE_API_KEY` | | | Gemini para embeddings (usada também como fallback da chave do LLM) |

Chaves gratuitas: [Groq](https://console.groq.com/keys) e [Google AI Studio](https://aistudio.google.com/apikey). Não requerem cartão de crédito.

## Execução

1. **LLM local (opcional):**

   ```bash
   ollama pull gemma2:2b
   ollama serve
   ```

2. **Backend** (a partir da raiz do projeto):

   ```bash
   uvicorn backend.app:app --reload
   ```

3. **Frontend:**

   ```bash
   cd frontend
   npm start
   ```

   A aplicação fica disponível em `http://localhost:4200`.

Para confirmar o fornecedor em uso, consulta `GET /health/llm` e `GET /health/embeddings`, ou vê os badges no canto superior da interface.

## Documentos

Os documentos são adicionados no **painel de Administração** da interface:

- **Upload de PDFs:** o ficheiro é guardado em `backend/data/`, extraído, dividido em chunks e indexado.
- **Scrapers:** páginas do site institucional podem ser recolhidas e indexadas a partir do mesmo painel.

Ao arrancar, o sistema não indexa automaticamente ficheiros copiados à mão para `backend/data/`. Com o índice vazio, o assistente responde a pedir que sejam adicionados documentos.

Cada provider de embeddings tem a sua própria coleção ChromaDB (`ipb_docs_v3_local` ou `ipb_docs_v3_gemini`), porque vetores de espaços diferentes não são comparáveis. Trocar de provider exige carregar os documentos nesse modo uma vez.

## Extração de PDFs

| Modo | Biblioteca | Quando é usado |
|---|---|---|
| `markdown` (defeito) | PyMuPDF4LLM | PDFs com prosa e tabelas |
| `table` | PyMuPDF `find_tables` | Horários e listas com uma identidade clara por linha |
| `text` | PyMuPDF `get_text` | PDFs simples, como fallback |

A escolha automática descarta o modo tabular quando os chunks resultantes são ilegíveis. Para forçar um modo:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"mode":"markdown"}' \
     http://localhost:8000/documents/NomeDoFicheiro.pdf/reextract
```

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Estado geral |
| GET | `/health/llm` | Fornecedor, modelo e disponibilidade do LLM |
| GET | `/health/embeddings` | Fornecedor, modelo, dimensão, coleção e contagem |
| POST | `/query` | Pergunta ao assistente |
| POST | `/upload` | Carregar PDF |
| GET | `/documents` | Listar documentos |
| GET | `/documents/{filename}/file` | Servir o PDF |
| PATCH | `/documents/{filename}/rename` | Renomear documento |
| POST | `/documents/{filename}/reextract` | Re-extrair com outro modo |
| DELETE | `/documents/{filename}` | Apagar documento |
| GET, POST | `/conversations` | Listar e criar conversas |
| GET, PATCH, DELETE | `/conversations/{id}` | Gerir uma conversa |
| GET, POST | `/admin/scrapers` | Gerir scrapers do site institucional |
| POST | `/admin/scrapers/{id}/run` | Executar um scraper |

A documentação interativa completa está em `http://localhost:8000/docs` (gerada pelo FastAPI).

## Estrutura

```
backend/
  app.py            Endpoints REST (FastAPI)
  answer.py         Composição da resposta via LLM
  retrieval.py      Pesquisa híbrida (ChromaDB + BM25)
  ingest.py         Carregamento de PDFs e texto
  chunk.py          Divisão em chunks
  pdf_tables.py     Extração tabular
  schema_store.py   Registo persistente de esquemas tabulares
  llm_client.py     Abstração unificada Ollama / Groq / Gemini
  vision.py         Extração de PDFs digitalizados
  conversations.py  Histórico de conversas
frontend/           Interface Angular 17
requirements.txt    Dependências Python
```

## Avaliação

Benchmark com **160 perguntas** em 8 categorias (20 por categoria), com um conjunto de respostas de referência.

**Recusa (anti-alucinação):** nas 40 perguntas sem resposta nos documentos (categorias "desconhecido" e "anti-alucinação"), o sistema recusou responder em **100%** dos casos.

**Recuperação** (os documentos certos aparecem nos resultados):

| Categoria | Recall@5 | Recall@20 | MRR |
|---|---|---|---|
| Factual | 0,90 | 1,00 | 0,72 |
| Restrição | 0,86 | 1,00 | 0,82 |
| Multi-hop | 0,72 | 0,98 | 1,00 |
| Conversacional | 0,80 | 0,90 | 0,54 |
| Listagem | 0,46 | 0,80 | 0,68 |

**Qualidade da resposta** (avaliada por um LLM juiz, 100 perguntas):

| Categoria | Correta | Parcial | Incorreta |
|---|---|---|---|
| Factual | 90% | 0% | 10% |
| Multi-hop | 40% | 50% | 10% |
| Listagem | 35% | 45% | 20% |
| Restrição | 25% | 25% | 50% |
| Conversacional | 5% | 0% | 95% |

Perguntas factuais diretas funcionam bem e a recusa é fiável. Perguntas com restrições, listagens longas, contagens e follow-ups conversacionais são os principais pontos a melhorar.

Para correr o benchmark: ver `benchmark/` (`run_benchmark.py`, `evaluate.py`, `judge.py`) e `backend/eval/`.

## Autor

Cláudio Azevedo · Engenharia Informática, IPB/ESTiG, 2026
