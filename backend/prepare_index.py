import os
from ingest import load_text_documents, scrape_ipb_page
from chunk import chunk_text
from retrieval import create_collection, index_documents


def build_index(text_paths, web_urls):
    collection = create_collection(persist_directory="backend/db", name="ipb_docs_v2")

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
    text_paths = ["data/sample.txt"]
    web_urls = ["https://www.ipb.pt"]

    if not any(os.path.exists(path) for path in text_paths):
        print("Aviso: nenhum arquivo de texto encontrado em backend/data/. Adicione arquivos .txt e execute novamente.")

    build_index(text_paths, web_urls)
