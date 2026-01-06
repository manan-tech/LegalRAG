
import os
import time
import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_community.document_loaders import UnstructuredPDFLoader
from langgraph.prebuilt import create_react_agent # NEW: Using LangGraph
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CPC_INDEX = os.path.join(BASE_DIR, "faiss_index_cpc")
ARMY_INDEX = os.path.join(BASE_DIR, "faiss_index_army_act")
TEST_CASE_PATH = os.path.join(BASE_DIR, "Case.pdf")

# Check if indexes exist
if not os.path.exists(CPC_INDEX) or not os.path.exists(ARMY_INDEX):
    print("❌ FAISS indexes not found. Please run data_ingestion.py first.")
    exit(1)

print("🚀 Starting LegalRAG Pipeline Evaluation with Metrics...")

# 1. Initialize Models
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY not found in .env")
    exit(1)

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Loading Vector Stores...")
cpc_vectorstore = FAISS.load_local(CPC_INDEX, embeddings, allow_dangerous_deserialization=True)
army_vectorstore = FAISS.load_local(ARMY_INDEX, embeddings, allow_dangerous_deserialization=True)

# 2. Define Tools
@tool
def retrieve_from_case_file(query: str) -> str:
    """Reads the uploaded case file."""
    if not os.path.exists(TEST_CASE_PATH):
        return "Error: Test Case PDF not found."
    loader = UnstructuredPDFLoader(TEST_CASE_PATH)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def retrieve_from_cpc(query: str) -> str:
    """Searches CPC sections."""
    retrieved_docs = cpc_vectorstore.similarity_search(query, k=3)
    return "\n\n".join(f"📘 Source Page: {doc.metadata.get('page', 'Unknown')}\n\n{doc.page_content}" for doc in retrieved_docs)

@tool
def retrieve_from_army_code(query: str) -> str:
    """Searches military law."""
    retrieved_docs = army_vectorstore.similarity_search(query, k=3)
    return "\n\n".join(f"🪖 Source Page: {doc.metadata.get('page', 'Unknown')}\n\n{doc.page_content}" for doc in retrieved_docs)

tools = [retrieve_from_case_file, retrieve_from_cpc, retrieve_from_army_code]

# 3. Define Agent (LangGraph)
system_prompt = (
    "You are an expert legal strategist. Your task is to generate suggestions for a defendant "
    "based on a case file and relevant legal codes. You MUST follow this exact plan:\n\n"
    "**Step 1:** Analyze the Case File using `retrieve_from_case_file`.\n"
    "**Step 2:** Find Relevant Law (CPC or Army Act).\n"
    "**Step 3:** Synthesize and Advise."
)

# Create the specific LangGraph agent
agent_executor = create_react_agent(llm, tools, prompt=system_prompt)

# 4. Define Test Data with Ground Truth Keywords
test_scenarios = [
    {
        "query": "Analyze the case and suggest a legal strategy.",
        "expected_keywords": ["petition", "army", "tribunal", "dismissal"],
        "min_citations": 1
    },
    {
        "query": "Which specific Army Act sections are relevant?",
        "expected_keywords": ["Section", "Act", "1950"],
        "min_citations": 2
    },
    {
        "query": "What is the procedure for appeal in this jurisdiction?",
        "expected_keywords": ["Appeal", "Court", "Time"],
        "min_citations": 1
    }
]

metrics_data = []

# 5. Run Evaluation Loop
results_log = "# LegalRAG Pipeline Evaluation Report\n\n"

for i, scenario in enumerate(test_scenarios, 1):
    query = scenario["query"]
    expected = scenario["expected_keywords"]
    
    print(f"\n--- Running Test Query {i}: {query} ---")
    start_time = time.time()
    
    try:
        # LangGraph invoke returns a dictionary with 'messages'
        response = agent_executor.invoke({"messages": [{"role": "user", "content": query}]})
        # Extract the content of the last message (AIMessage)
        output = response["messages"][-1].content
    except Exception as e:
        output = str(e)
        
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    
    # --- CALCULATE METRICS ---
    # 1. Latency
    latency = duration
    
    # 2. Keyword Recall (Simple Accuracy Proxy)
    matches = sum(1 for keyword in expected if keyword.lower() in output.lower())
    recall_score = (matches / len(expected)) * 100 if expected else 0
    
    # 3. Citation Count (Groundedness Proxy)
    # Looking for patterns like "Page:", "Section", "Source:"
    citation_matches = re.findall(r'(Source Page|Section \d+|Act \d+)', output, re.IGNORECASE)
    citation_count = len(citation_matches)
    
    # Log Data
    metrics_data.append({
        "Test_ID": i,
        "Query": query,
        "Latency_Seconds": latency,
        "Keyword_Recall_Msg": f"{matches}/{len(expected)}",
        "Recall_Percent": recall_score,
        "Citation_Count": citation_count
    })
    
    results_log += f"## Test {i}: {query}\n"
    results_log += f"- **Latency:** {latency}s\n"
    results_log += f"- **Recall:** {recall_score}% (Found {matches} of {len(expected)} keywords)\n"
    results_log += f"- **citations:** {citation_count}\n"
    results_log += f"\n**Agent Response:**\n{output}\n\n---\n"

# 6. Save Report and CSV
with open("pipeline_evaluation_report.md", "w") as f:
    f.write(results_log)

df = pd.DataFrame(metrics_data)
df.to_csv("evaluation_metrics.csv", index=False)

# Calculate Macro Averages
avg_recall = df["Recall_Percent"].mean()
avg_latency = df["Latency_Seconds"].mean()
total_citations = df["Citation_Count"].sum()

print(f"\n✅ Evaluation Complete.")
print(f"📊 Average Recall (Accuracy Proxy): {avg_recall:.2f}%")
print(f"⏱️ Average Latency: {avg_latency:.2f}s")
print(f"📚 Total Citations Generated: {total_citations}")
print("Results saved to 'pipeline_evaluation_report.md' and 'evaluation_metrics.csv'")
