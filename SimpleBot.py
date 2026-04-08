# app.py

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

import json
import os

chromaloc = "chroma_db"
jsonloc = "split_docs.json"


def load_split_docs(file_path=jsonloc):
    with open(file_path, "r", encoding="utf-8") as f:
        split_docs = json.load(f)

    return [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in split_docs
    ]


# Dense retrieval with Chroma
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    persist_directory=chromaloc,
    embedding_function=embeddings
)

dense_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)


# Sparse retrieval with BM25
split_docs = load_split_docs()

bm25_retriever = BM25Retriever.from_documents(split_docs)
bm25_retriever.k = 4


#Prompts
prompt = PromptTemplate(
    template="""You are a question-answering model.
You can only answer questions based on the context provided above in docs from Georgia State University's official website.
You can only use information present to answer the question.
If the answer is not explicitly stated, respond with "I don't know."
If the question is not related to the context, respond with "I don't know."
If the question is about a different university, respond with "I don't know."
If the question contains any verbal abuse or harmful content, respond with "I don't know."
If the question is about anything illegal or unethical, respond with "I don't know."
If the question is about anything political, respond with "I don't know."
If the question is beyond context, respond with "I don't know."

Documents:
{documents}

Query:
{question}

Answer (max 3 sentences):
""",
    input_variables=["question", "documents"],
)

langModel = ChatOllama(model="llama3.1", temperature=0)
ragChain = prompt | langModel | StrOutputParser()


class Application:
    def __init__(self, dense_retriever, bm25_retriever, ragChain):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.ragChain = ragChain

    def hybrid_retrieval(self, query):
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)

        combined_docs = {
            doc.page_content: doc
            for doc in dense_docs + sparse_docs
        }

        return list(combined_docs.values())

    def run(self, query):
        if not query:
            return "Please enter a question."

        docs = self.hybrid_retrieval(query)
        if not docs:
            return "I don't know based on the docs provided."

        context = "\n\n".join(doc.page_content for doc in docs)

        response = self.ragChain.invoke({
            "question": query,
            "documents": context
        })
        return response
    
RAG = Application(dense_retriever, bm25_retriever, ragChain)

tests = [
    "What are the admission requirements for the Computer Science BS program?",
    "How many credit hours are required for the Computer Science MS program?",
    "What core courses are required for the Computer Science undergraduate program?",
    "What is the minimum GPA requirement for the Computer Science program to graduate?",
    "What is CSC 1301?"
]

for query in tests:
    answer = RAG.run(query)
    print("Q:", query)
    print("A:", answer)
    print()