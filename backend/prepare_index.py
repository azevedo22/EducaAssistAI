"""Indexação a partir da linha de comandos (alternativa ao painel de Administração).

Correr a partir da raiz do projeto:
    python -m backend.prepare_index

Usa a mesma coleção que o backend (ipb_docs_v3_<EMBED_PROVIDER>) e a mesma
pasta (backend/db), pelo que os documentos ficam visíveis na aplicação.
"""
import os

from backend.ingest import load_text_documents, scrape_ipb_page
from backend.chunk import chunk_text
from backend.retrieval import collection_name_for_provider, create_collection, index_documents


def build_index(text_paths, web_urls):
    collection = create_collection(
        persist_directory="backend/db",
        name=collection_name_for_provider("ipb_docs_v3"),
    )

    texts = []
    metadatas = []
    ids = []
    counter = 0

    for document in load_text_documents(text_paths):
        source = document["source"]
        for chunk in chunk_text(document["text"]):
            texts.append(chunk)
            metadatas.append({"source": source})
            ids.append(f"txt-{counter}")
            counter += 1

    for url in web_urls:
        text = scrape_ipb_page(url)
        source = url
        for chunk in chunk_text(text):
            texts.append(chunk)
            metadatas.append({"source": source})
            ids.append(f"web-{counter}")
            counter += 1

    index_documents(collection, texts, metadatas, ids)
    print(f"Index criado com {len(texts)} chunks.")


if __name__ == "__main__":
    text_paths = ["backend/data/sample.txt"]
    web_urls = ["https://www.ipb.pt"]

    if not any(os.path.exists(path) for path in text_paths):
        print("Aviso: nenhum arquivo de texto encontrado em backend/data/. Adicione arquivos .txt e execute novamente.")

    build_index(text_paths, web_urls)
