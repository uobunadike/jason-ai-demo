import os
from dotenv import load_dotenv  # Load environment variables from a .env file

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models import ChatOpenAI
from functools import lru_cache

# ✅ Safety check to ensure vector store exists
if not os.path.exists("./chroma_db"):
    raise FileNotFoundError("Missing ./chroma_db. Please generate the vector store first.")

@lru_cache(maxsize=1)
def get_vector_store():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
    )

def run(query: str, model_type: str = "ollama", model_name: str = "llama3.1") -> str:
    db = get_vector_store()

    if model_type == "ollama":
        llm = OllamaLLM(model=model_name, temperature=0.3)
    else:
        llm = ChatOpenAI(model=model_name, temperature=0.5, openai_api_key=openai_api_key)

    if query.strip().lower() in ["hi", "hello", "hey", "what's up?", "how are you?"]:
        return llm.predict(query)

    template = """You are an AI inventory insights assistant.

Based on the extracted data, answer the question concisely using the format below:

- **Summary**: 1-2 sentence answer to the question.
- **Root Cause**: Short explanation if there’s a risk or issue.
- **Business Impact**: Estimated dollar or operational impact (if applicable).
- **Recommendation**: Actionable next step (if possible).

Context:
{context}


Question: {question}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain.invoke({"query": query})['result']

