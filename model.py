import os
from dotenv import load_dotenv  # Load environment variables from a .env file

# Load environment variables
load_dotenv()
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION"

# LangChain imports
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from functools import lru_cache

# ✅ Safety check to ensure vector store exists
if not os.path.exists("./chroma_db"):
    raise FileNotFoundError("Missing ./chroma_db. Please generate the vector store first.")

@lru_cache(maxsize=1)
def get_vector_store():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=AzureOpenAIEmbeddings(
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
           deployment=azure_embedding_deployment,
           model=azure_embedding_deployment,
         api_version=azure_api_version
        )
    )

def run(query: str, model_type: str = "ollama", model_name: str = "llama3.1") -> str:
    db = get_vector_store()

    if model_type == "ollama":
        llm = OllamaLLM(model=model_name, temperature=0.3)
    else:
        llm = AzureChatOpenAI(
            api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    deployment_name=azure_chat_deployment,
    model=azure_chat_deployment,
    api_version=azure_api_version,
    temperature=0.5
        )

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
