# app.py

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load saved vectorstore
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}, search_type="mmr")

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
    def __init__(self, retriever, ragChain):
        self.retriever = retriever
        self.ragChain = ragChain

    def run(self, query):
        if not query:
            return "Please enter a question."

        docs = self.retriever.invoke(query)
        if not docs:
            return "I don't know based on the docs provided."

        context = "\n\n".join([doc.page_content for doc in docs])

        response = self.ragChain.invoke({
            "question": query,
            "documents": context
        })
        return response

RAG = Application(retriever, ragChain)

tests = [
    "What are the admission requirements for the Computer Science BS program?",
    "How many credit hours are required for the Computer Science MS program?",
    "What core courses are required for the Computer Science undergraduate program?",
    "What is the minimum GPA requirement for the Computer Science program to graduate?",
]

for query in tests:
    answer = RAG.run(query)
    print("Q:", query)
    print("A:", answer)
    print()