from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

DATA_PATH = "data"
DB_PATH = "vectorstores/db"

def create_vector_db():
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("Vector DB already exists. Skip.")
        return

    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Processed {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
            "Ⅰ.",
            "Ⅱ.",
            "Ⅲ.",
            "1.",
            "2.",
            "(1)",
            "(2)",
            "다.",
            "함.",
            "."
        ]
    )


    texts = text_splitter.split_documents(documents)

    print(f"총 청크 개수: {len(texts)}")
    print(texts[0].page_content)

    for doc in texts:
        doc.metadata["source"] = os.path.basename(doc.metadata.get("source", ""))
        doc.metadata["page"] = doc.metadata.get("page", None)



    embedding = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 8}
    )

    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=DB_PATH
    )

    print("Vector DB created")

if __name__ == "__main__":
    os.makedirs(DB_PATH, exist_ok=True)
    create_vector_db()