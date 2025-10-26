from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain.tools import tool
import faiss
from dotenv import load_dotenv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()


llm = init_chat_model("moonshotai/kimi-k2-instruct-0905", model_provider="groq")
embeddings = OllamaEmbeddings(model="embeddinggemma:300m")

embeddings = OllamaEmbeddings(model="embeddinggemma:300m")
vectorstore = FAISS.load_local("faiss_index_Case", embeddings, allow_dangerous_deserialization=True)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve the most relevant context passages from the FAISS vector database
    based on a user's query. Returns both the concatenated text and document metadata.
    """
    retrieved_docs = vectorstore.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

system_prompt = (
    "You are an expert lawyer. Your task is to answer the user's query "
    "by synthesizing information *only* from the case file, which you "
    "access using the 'retrieve_context' tool."
    "\n\n"
    "**Your Strategy:**"
    "1.  **Deconstruct:** Break down the user's complex query into smaller, "
    "    specific questions."
    "2.  **Query:** Use the `retrieve_context` tool for *each* specific question. "
    "    (e.g., 'key background facts', 'primary legal issues', 'plaintiff arguments', "
    "    'defendant arguments', 'final holding or conclusion')."
    "3.  **Synthesize:** Once you have all the retrieved context, combine it into a "
    "    single, coherent answer. "
    "4.  **Cite:** Do not add any information you did not retrieve. "
    "    Do not answer from your own knowledge."
)

agent = create_agent(llm, tools, system_prompt=system_prompt)

user_query = (
    "Provide a summary of the case, identifying the parties, the key background facts, the primary legal issues, the main arguments from both sides, and the court's final holding or conclusion."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": user_query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()