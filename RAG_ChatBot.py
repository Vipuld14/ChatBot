# app.py

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
import json
import os

faissloc = "faiss_db"
jsonloc = "split_docs.json"


def load_split_docs(file_path=jsonloc):
    with open(file_path, "r", encoding="utf-8") as f:
        split_docs = json.load(f)

    return [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in split_docs
    ]


# Dense retrieval with FAISS
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.load_local(
    faissloc,
    embeddings,
    allow_dangerous_deserialization=True
)


dense_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)

compressor = FlashrankRerank(top_n=3)
reranker = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=dense_retriever
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

Answer (max 3 sentences). End your responses with a new line with a Source attribution in the format [Source: URL] where URL is the source of the information from the documents provided. Make sure the source URl doesnt exit the text box.

""",
    input_variables=["question", "documents"],
)

langModel = ChatOllama(model="llama3.2", temperature=0)
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
        allDocs = list(combined_docs.values())

        reranked_docs = compressor.compress_documents(allDocs, query)
        return reranked_docs if reranked_docs else allDocs

    def run(self, query):
        if not query:
            return "Please enter a question."

        docs = self.hybrid_retrieval(query)
        if not docs:
            return "I don't know based on the docs provided."

        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'Unknown')}] {doc.page_content}" for doc in docs)

        response = self.ragChain.invoke({
            "question": query,
            "documents": context
        })
        return response
    
RAG = Application(dense_retriever, bm25_retriever, ragChain)

