from altair import Y
import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.agents import create_agent
from dotenv import load_dotenv
import tempfile
import os
import re
from datetime import datetime

st.set_page_config(
    page_title="Explainable Legal Agent",
    page_icon="⚖️",
    layout="wide"
)

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@st.cache_resource
def get_llm(api_key):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key
    )

@st.cache_resource
def get_embeddings():
    try:
        embeddings = OllamaEmbeddings(model="embeddinggemma:300m")
        embeddings.embed_query("test")
        return embeddings
    except Exception as e:
        st.error(f"Ollama connection failed: {e}. Please ensure Ollama is running.", icon="🚨")
        st.stop()

@st.cache_resource
def load_vector_stores(_embeddings):
    try:
        cpc_vectorstore = FAISS.load_local("faiss_index_cpc", _embeddings, allow_dangerous_deserialization=True)
        army_vectorstore = FAISS.load_local("faiss_index_army_act", _embeddings, allow_dangerous_deserialization=True)
        return cpc_vectorstore, army_vectorstore
    except Exception as e:
        st.error(f"Failed to load vector stores: {e}", icon="🚨")
        st.info("Please make sure the 'faiss_index_cpc' and 'faiss_index_army_act' directories exist.")
        st.stop()


def main():
    st.title("Explainable Legal Agent ⚖️")
    st.subheader("AI-Generated Strategic Roadmap for the Defendant")

    with st.sidebar:
        st.header("Configuration")
        st.info("Ensure Ollama is running locally to use the embedding model.")
        
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
        
        if not groq_api_key:
            st.warning("Please enter your Groq API Key to proceed.")
            st.stop()

        uploaded_case = st.file_uploader("📄 Upload Case File (PDF)", type=["pdf"])
        if uploaded_case is not None:
            temp_case_path = os.path.join(tempfile.gettempdir(), uploaded_case.name)
            with open(temp_case_path, "wb") as f:
                f.write(uploaded_case.read())
            st.success(f"✅ Case file uploaded: {uploaded_case.name}")
        else:
            st.warning("Please upload a case file to continue.")
            st.stop()

    with st.spinner("Loading models and vector stores..."):
        llm = get_llm(groq_api_key)
        embeddings = get_embeddings()
        cpc_vs, army_vs = load_vector_stores(embeddings)

    # ---- TOOLS ----
    @tool
    def retrieve_from_case_file(query: str) -> str:
        """Reads the uploaded case file and returns its contents."""
        loader = PyPDFLoader(temp_case_path)
        docs = loader.load()
        return "\n\n".join(doc.page_content for doc in docs)

    @tool
    def retrieve_from_cpc(query: str) -> str:
        """Searches CPC sections relevant to a civil case."""
        retrieved_docs = cpc_vs.similarity_search(query, k=3)
        return "\n\n".join(
            f"📘 Source Page: {doc.metadata.get('page', 'Unknown')}\n\n{doc.page_content}"
            for doc in retrieved_docs
        )

    @tool
    def retrieve_from_army_code(query: str) -> str:
        """Searches military law sections if applicable."""
        retrieved_docs = army_vs.similarity_search(query, k=3)
        return "\n\n".join(
            f"🪖 Source Page: {doc.metadata.get('page', 'Unknown')}\n\n{doc.page_content}"
            for doc in retrieved_docs
        )

    tools = [retrieve_from_case_file, retrieve_from_cpc, retrieve_from_army_code]

    system_prompt = (
        "You are an expert legal strategist. Your task is to generate suggestions for a defendant "
        "based on a case file and relevant legal codes. You MUST follow this exact plan:\n\n"
        "**Step 1: Analyze the Case File.** Use `retrieve_from_case_file` first to understand the facts, arguments, and holding.\n"
        "Determine if it's a CIVIL or CRIMINAL case and if it involves any military personnel.\n\n"
        "**Step 2: Find Relevant Law.**\n"
        "- If CIVIL → use `retrieve_from_cpc`.\n"
        "- If MILITARY → use `retrieve_from_army_code`.\n"
        "- Never use the same tool more than 3 times.\n\n"
        "**Step 3: Synthesize and Advise.** Summarize procedural timeline, list completed vs pending actions, "
        "and suggest actionable next steps for the defendant."
    )

    agent = create_agent(llm, tools, system_prompt=system_prompt)

    def get_agent_output(agent_to_run, query):
        final_output = ""
        trace_logs = []
        tool_usage = {t.name: 0 for t in tools}
        MAX_CALLS = 3

        try:
            for event in agent_to_run.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            ):
                if "messages" in event and event["messages"]:
                    final_output = event["messages"][-1].content

                if event.get("type") == "tool":
                    tool_name = event.get("tool", "unknown_tool")

                    if tool_name in tool_usage:
                        if tool_usage[tool_name] >= MAX_CALLS:
                            trace_logs.append({
                                "tool": tool_name,
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "input": event.get("input", ""),
                                "output": "⚠️ Skipped - Max call limit reached"
                            })
                            continue

                        tool_usage[tool_name] += 1
                        trace_logs.append({
                            "tool": tool_name,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "input": event.get("input", ""),
                            "output": event.get("output", "")
                        })

            return final_output, trace_logs

        except Exception as e:
            return f"Error during agent execution: {e}", []

    user_query = (
        "Please analyze the case file, find applicable CPC or Army Code sections if any, "
        "and generate a set of suggestions for the defendant."
    )

    if st.button("🚀 Run Legal Agent", type="primary"):
        with st.spinner("Analyzing case..."):
            final_answer, trace_logs = get_agent_output(agent, user_query)

        st.markdown("## 🧭 Final Roadmap")
        st.markdown(final_answer)

        st.divider()
        st.markdown("## 🧩 Agent Trace (Tool Calls & Observations)")
        if trace_logs:
            for i, log in enumerate(trace_logs, 1):
                with st.expander(f"🔧 Step {i} — {log['tool']} ({log['timestamp']})"):
                    st.markdown(f"**Input:** `{log['input']}`")
                    st.markdown(f"**Output:**\n\n{log['output'][:1000]}{'...' if len(log['output'])>1000 else ''}")
        else:
            st.info("No tool activity recorded.")

if __name__ == "__main__":
    main()