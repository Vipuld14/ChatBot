#Import necessary libraries

#Libararies for Documets Loading and Splitting
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import re
from langchain_chroma import Chroma

import json
import os

import shutil


# Document Loading
urls = [
    "https://catalogs.gsu.edu/preview_program.php?catoid=4&poid=1159",
    "https://catalogs.gsu.edu/preview_program.php?catoid=43&poid=12713",
    "https://catalogs.gsu.edu/preview_entity.php?catoid=43&ent_oid=2982",
    "https://www.gsu.edu/program/computer-science-bs/?utm_source=pltitle&utm_medium=cas&utm_content=bs&utm_campaign=program_explorer",
    "https://www.gsu.edu/program/computer-science-ms/?utm_source=pltitle&utm_medium=cas&utm_content=ms&utm_campaign=program_explorer",
    "https://catalogs.gsu.edu/content.php?catoid=42&navoid=5496",
    "https://catalogs.gsu.edu/content.php?catoid=42&navoid=5496#3010-general-information",
    "https://communication.gsu.edu/document/ma-handbook/?wpdmdl=4945&refresh=5faed98232b1d1605294466",
    "https://csds.gsu.edu/?wpdmdl=4939&ind=1620936669195"
]

chromaLocation = "chroma_db"
JsonPath = "split_docs.json"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def load_and_clean_documents():
    documents = [WebBaseLoader(url).load() for url in urls]
    document_list = [doc for subset in documents for doc in subset]

    cleaned_docs = []
    for doc in document_list:
        cleaned_content = clean_text(doc.page_content)
        cleaned_docs.append(
            Document(page_content=cleaned_content, metadata=doc.metadata)
        )

    return cleaned_docs

def split_documents(cleaned_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_documents(cleaned_docs)

def save_split_docs(split_docs, file_path = JsonPath):
    serialized_docs = []
    for doc in split_docs:
        serialized_docs.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serialized_docs, f, indent=4, ensure_ascii=False)

def build_and_save_vectorstore():
    cleaned_docs = load_and_clean_documents()
    split_docs = split_documents(cleaned_docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if not os.path.exists(chromaLocation):
        shutil.rmtree(chromaLocation)

    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=chromaLocation
    )

    save_split_docs(split_docs)

    print("Vector store built and saved successfully.")

if __name__ == "__main__":
    build_and_save_vectorstore()