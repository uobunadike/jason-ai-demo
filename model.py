import sys

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass


import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from functools import lru_cache

# Add these lines at the VERY TOP of your Python file

# Now import LangChain/Chroma components
from langchain_community.vectorstores import Chroma


from pathlib import Path
#print("Looking for .env at:", Path(".env").resolve())

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)


#print("DEBUG - from getenv:", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"))

# Azure configs from .env
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# 🔐 Debugging log
#print("🔐 Using Azure config:")
#print("Endpoint:", endpoint)
#print("Chat deployment:", chat_deployment)
#print("Embed deployment:", embedding_deployment)
#print("Version:", api_version)

# Safety check for Chroma vector store
if not os.path.exists("./chroma_db"):
    raise FileNotFoundError("Missing ./chroma_db. Please generate it first.")

@lru_cache(maxsize=1)
def get_vector_store():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=AzureOpenAIEmbeddings(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            deployment=embedding_deployment
        )
    )

def run(query: str, model_type="azure", model_name="gpt-35-turbo") -> str:
    db = get_vector_store()

    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        deployment_name=chat_deployment,
        temperature=0.5
    )

    prompt_template = """You are an AI inventory insights assistant.

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
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain.invoke({"query": query})['result']