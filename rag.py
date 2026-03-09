import os

os.environ["USER_AGENT"] = "GSU-Chatbot/1.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.documents import Document

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

VECTORSTORE_PATH = "vectorstore/gsu_vectors.json"

def clean_text(text):
    return text.strip()

def build_documents():
    documents = [WebBaseLoader(url).load() for url in urls]
    document_list = [doc for subset in documents for doc in subset]

    cleaned_docs = []
    for doc in document_list:
        cleaned_content = clean_text(doc.page_content)
        cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return text_splitter.split_documents(cleaned_docs)

def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    os.makedirs("vectorstore", exist_ok=True)

    if os.path.exists(VECTORSTORE_PATH):
        print("Loading saved vector store...")
        return SKLearnVectorStore(
            embedding=embeddings,
            persist_path=VECTORSTORE_PATH,
            serializer="json"
        )

    print("Building vector store for the first time...")
    document_split = build_documents()

    vectorstore = SKLearnVectorStore.from_documents(
        documents=document_split,
        embedding=embeddings,
        persist_path=VECTORSTORE_PATH,
        serializer="json"
    )

    vectorstore.persist()
    print("Vector store saved.")

    return vectorstore

def build_rag():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

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

    lang_model = ChatOllama(model="llama3.1", temperature=0)
    rag_chain = prompt | lang_model | StrOutputParser()

    return retriever, rag_chain


class Application:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, query):
        if not query.strip():
            return "Please enter a question."

        docs = self.retriever.invoke(query)
        if not docs:
            return "I don't know based on the docs provided."

        context = "\n\n".join([doc.page_content for doc in docs])

        response = self.rag_chain.invoke({
            "question": query,
            "documents": context
        })

        return response


retriever, rag_chain = build_rag()
RAG = Application(retriever, rag_chain)
