from altair import Y
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_community.document_loaders import UnstructuredPDFLoader
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage, AIMessage
from dotenv import load_dotenv
import tempfile
import os
import datetime
import re

st.set_page_config(page_title="Explainable Legal Agent", page_icon="⚖️", layout="wide")
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@st.cache_resource
def get_llm(api_key):
    return ChatGroq(model="openai/gpt-oss-120b", api_key=api_key)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vector_stores(_embeddings):
    cpc_vectorstore = FAISS.load_local("faiss_index_cpc", _embeddings, allow_dangerous_deserialization=True)
    army_vectorstore = FAISS.load_local("faiss_index_army_act", _embeddings, allow_dangerous_deserialization=True)
    return cpc_vectorstore, army_vectorstore

def main():
    st.title("Explainable Legal Agent ⚖️")
    st.subheader("AI-Generated Strategic Roadmap for the Defendant")

    with st.sidebar:
        st.header("Configuration")
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
            # Set a dummy path to avoid errors before upload, though st.stop handles it.
            temp_case_path = ""

    with st.spinner("Loading models and vector stores..."):
        llm = get_llm(groq_api_key)
        embeddings = get_embeddings()
        cpc_vs, army_vs = load_vector_stores(embeddings)

    @tool
    def retrieve_from_case_file(query: str) -> str:
        """Reads the uploaded case file and returns its contents."""
        if not os.path.exists(temp_case_path):
             return "Case file not found."
        loader = UnstructuredPDFLoader(temp_case_path)
        docs = loader.load()
        return "\n\n".join(doc.page_content for doc in docs)

    @tool
    def retrieve_from_cpc(query: str) -> str:
        """Searches CPC sections relevant to a civil case."""
        retrieved_docs = cpc_vs.similarity_search(query, k=3)
        return "\n\n".join(f"📘 Source Page: {doc.metadata.get('page', 'Unknown')}\n\n{doc.page_content}" for doc in retrieved_docs)

    @tool
    def retrieve_from_army_code(query: str) -> str:
        """Searches military law sections if applicable."""
        retrieved_docs = army_vs.similarity_search(query, k=3)
        return "\n\n".join(f"🪖 Source Page: {doc.metadata.get('page', 'Unknown')}\n\n{doc.page_content}" for doc in retrieved_docs)

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

    # Replaced create_agent (Legacy) with create_react_agent (LangGraph)
    agent_graph = create_react_agent(llm, tools, prompt=system_prompt)

    def get_agent_output(agent_to_run, query):
        final_output = ""
        trace_logs = []
        
        try:
            # LangGraph: We invoke the graph with the initial state
            response = agent_to_run.invoke({"messages": [{"role": "user", "content": query}]})
            
            # Extract Messages history to build Trace Logs
            messages = response.get("messages", [])
            
            # The last message is the final answer
            if messages:
                raw_output = messages[-1].content
                # Fix Issue 1: Remove <br> tags
                final_output = re.sub(r'<br\s*/?>', '\n', raw_output, flags=re.IGNORECASE)
            
            # Iterate through messages to find Tool Calls matches
            # Pattern: AIMessage (with tool_calls) -> ToolMessage (with artifact/content)
            for i, msg in enumerate(messages):
                if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name')
                        tool_args = tool_call.get('args')
                        tool_id = tool_call.get('id')
                        
                        # Find corresponding ToolMessage
                        tool_output = "No output found."
                        # Scan forward for the ToolMessage with this tool_call_id
                        for next_msg in messages[i+1:]:
                            if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id == tool_id:
                                tool_output = next_msg.content
                                break
                        
                        trace_logs.append({
                            "tool": tool_name,
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "input": str(tool_args),
                            "output": tool_output
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
            final_answer, trace_logs = get_agent_output(agent_graph, user_query)

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