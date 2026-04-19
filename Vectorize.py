#Import necessary libraries

#Libararies for Documets Loading and Splitting
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import re

import json
import os
import json5
from langchain_community.vectorstores import FAISS
import shutil


faissLocation = "faiss_db"
JsonPath = "split_docs.json"
SourcesPath = "sources.json"

# Document Loading
def load_sources(file_path=SourcesPath):
    if not os.path.exists(file_path):
        return []
    if os.path.getsize(file_path) == 0:
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    content = re.sub(r',\s*(\]|\})', r'\1', content)
    try:
        urls = json5.loads(content)
        return urls
    except Exception as e:
        print(f"Error decoding JSON: {e}")
        return []
def save_sources(sources, file_path=SourcesPath):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=4, ensure_ascii=False)



import gc
gc.collect() 
if os.path.exists(faissLocation):
    shutil.rmtree(faissLocation, ignore_errors=True)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def load_and_clean_documents():

    urls = load_sources()
    if not urls:
        raise ValueError("No URLs found in sources file.")
    
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
        chunk_size=400,
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

    if os.path.exists(faissLocation):
        shutil.rmtree(faissLocation, ignore_errors=True)

    vectorStore = FAISS.from_documents(
        split_docs,
        embeddings
    )
    vectorStore.save_local(faissLocation)

    save_split_docs(split_docs)

    print("Vector store built and saved successfully.")

if __name__ == "__main__":
    build_and_save_vectorstore()