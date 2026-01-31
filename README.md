# LegalRAG: Agentic AI for Explainable Legal Strategy ⚖️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://explainable-legal-rag.streamlit.app/)
[![Powered by LangGraph](https://img.shields.io/badge/Powered%20by-LangGraph-blue)](https://langchain-ai.github.io/langgraph/)
[![Ingestion by Unstructured](https://img.shields.io/badge/Ingestion-Unstructured.io-purple)](https://unstructured.io/)

> **Live Demo:** [https://explainable-legal-rag.streamlit.app/](https://explainable-legal-rag.streamlit.app/)

## 📜 Overview
**LegalRAG** is an advanced **Agentic AI** system designed to act as an intelligent co-pilot for legal defense strategy. Unlike standard chatbots that simply "guess" answers or basic RAG systems that blindly retrieve text, LegalRAG uses a **Reason+Act (ReAct)** cognitive architecture.

It autonomously analyzes case files, identifies the correct jurisdiction (Civil vs. Military), selects the appropriate legal statutes using discrete tools, and generates a procedurally sound, step-by-step legal roadmap.

---

## 🚀 Key Features

### 1. 🧠 Agentic Orchestration (LangGraph)
Powered by **LangGraph**, the system doesn't follows a linear script. It thinks before it acts.
*   **Dynamic Routing:** Automatically detects if a case involves Military Personnel and routes queries to the `Army Act` instead of Civil Code.
*   **Chain of Thought:** You can see the agent's reasoning process (e.g., *"The user is asking about a court-martial. I should check Section 63 of the Army Act."*).

### 2. 📄 High-Fidelity Ingestion (Unstructured.io)
Legal documents are complex—filled with multi-column layouts, tables, and marginalia.
*   **Layout Awareness:** We use **`UnstructuredPDFLoader`** to parse PDFs, ensuring that tabular data (like schedules of fines or dates) is preserved as structured information, not jumbled text.
*   **Semantic Chunking:** Text is split in a way that preserves the meaning of legal clauses.

### 3. 🛠️ specialized Tool Use
The agent has access to a secure "Toolkit" to prevent hallucination:
*   `retrieve_from_case_file`: Reads the specific facts of the uploaded PDF case.
*   `retrieve_from_cpc`: Searches the *Code of Civil Procedure* (CPC) 1908.
*   `retrieve_from_army_code`: Searches the *Army Act, 1950* and Rules.

### 4. 🔍 Explainability & Traceability
Trust is critical in law.
*   **Trace Logs:** The sidebar displays the exact active "Thought Process" and JSON outputs of every tool call.
*   **Citations:** Every claim is backed by a specific Section or Page Number from the source document.

---

## 🆚 Comparison: Why Agentic AI?

| Feature | Standard LLM (ChatGPT/GPT-4) | Standard RAG | **LegalRAG (Agentic)** |
| :--- | :--- | :--- | :--- |
| **Data Source** | Training Data (Often Outdated) | Static Document Search | **Dynamic Tool Selection** |
| **Reasoning** | Implicit / Hazy | None (Keyword Matching) | **Explicit Multi-Step (ReAct)** |
| **Hallucination** | High Risk (Invents Laws) | Medium (Wrong Context) | **Near Zero (Grounded)** |
| **Input Parsing** | Text Paste (Loses Formatting) | Basic PDF Readers | **Unstructured.io (Layout Aware)** |
| **Transparency** | Black Box | "Sources" Link | **Full Execution Trace** |

---

## 📊 Evaluation Results
We rigorously tested the pipeline against complex, multi-jurisdictional synthetic scenarios.

*   **100% Keyword Recall:** The agent successfully identified all key legal concepts (e.g., "Section 63", "Tribunal", "Appeal Procedure") in every test case.
*   **100% Routing Accuracy:** It correctly distinguished when to apply Military Law vs. Civil Law in 3/3 complex test scenarios.
*   **Latency:** ~100s average response time. (We prioritize *deep reasoning correctness* over speed).

---

## 🛠️ Installation & Local Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/LegalRAG.git
    cd LegalRAG
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set API Keys**
    Create a `.env` file:
    ```bash
    GROQ_API_KEY=your_groq_api_key_here
    HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
    ```

4.  **Ingest Data** (Build Vector Store)
    ```bash
    python LegalRAG/data_ingestion.py
    ```

5.  **Run the App**
    ```bash
    streamlit run LegalRAG/app.py
    ```

---

## 🧪 Running Investigations
To run the automated evaluation suite:
```bash
python LegalRAG/evaluate_pipeline.py
```
This will generate a `pipeline_evaluation_report.md` with detailed performance metrics.

---

## 👨‍💻 Tech Stack
*   **Orchestration:** LangChain, LangGraph
*   **LLM:** Llama 3 / GPT-OSS (via Groq)
*   **Vector Store:** FAISS
*   **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
*   **ETL/Ingestion:** Unstructured.io
*   **Frontend:** Streamlit

## Contributors
*  **Parthiv Godrihal**
*  **Nilay Jain**
*  **Yuvan Jain**
