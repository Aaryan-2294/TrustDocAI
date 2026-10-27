# TrustDoc AI — Live Demonstration Guide (5–8 Minutes)

A concise, step-by-step presentation script showing the complete TrustDoc AI workflow: Ingestion, Evidence-Grounded Ask, Honest Abstention, Documentation Generation, and Automated AST Verification with Trust Scoring.

---

## 1. Pre-Demo Checklist & Environment Verification

Before presenting, ensure local services are running on the Windows machine:

1. **Virtual Environment**:
   - Python 3.12 virtual environment located at `.venv\Scripts\python.exe`.
2. **Local AI Services (CPU-Compatible)**:
   - Ollama server running at `http://127.0.0.1:11434` with model `qwen2.5-coder:1.5b`.
   - Embedder (`sentence-transformers/all-MiniLM-L6-v2`) and Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) cached under `E:\AI\Cache\HuggingFace\hub`.
3. **Streamlit UI**:
   - Web application accessible at `http://localhost:8501`.

### Verification Commands (PowerShell)
```powershell
# 1. Verify Ollama daemon and model availability
Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags"

# 2. Verify Streamlit application response
Invoke-RestMethod -Uri "http://localhost:8501" -Method Head
```

---

## 2. Launch Commands (If Services Are Not Running)

If the background servers are not already active, start them in order:

### Terminal 1: Start Ollama Daemon
```powershell
$env:OLLAMA_MODELS = "E:\AI\Models\Ollama"
& "E:\Apps\Ollama\ollama.exe" serve
```

### Terminal 2: Start Streamlit Application
```powershell
Set-Location C:\Users\Dell\Desktop\projects\TrustdocAI\trustdoc-ai
$env:HF_HOME = "E:\AI\Cache\HuggingFace"
$env:HF_HUB_CACHE = "E:\AI\Cache\HuggingFace\hub"
.\.venv\Scripts\python.exe -m trustdoc ui --port 8501
```
Open a browser to `http://localhost:8501`.

---

## 3. Click-by-Click Live Presentation Sequence

> **Testing & Validation Note**: The underlying API services, database queries, and data workflows described below were programmatically executed and validated using the real project environment and local models. However, because end-to-end browser automation drivers (e.g. Playwright or Selenium) are not installed in this environment, the click sequence below is documented directly from the verified Streamlit component code and has not been rehearsed via automated browser drivers.

### Step 1: System Overview & Project Selection (~1 min)
1. **Action**: Open the browser to `http://localhost:8501`.
2. **Action**: In the sidebar selectbox (**Select Project**), select:
   `retry-task-queue (id: 1)`
3. **What to point out on screen**:
   - **Project Metrics**: `Sources: 5`, `Entities: 3`, `Chunks: 19` (reflecting the base sources `workers.py` and `READme.md`, along with stored generated documentation artifacts).
   - **System & Configuration Expander**: Expand to show `LLM Provider: ollama`, `LLM Model: qwen2.5-coder:1.5b`, `Embedding Model: sentence-transformers/all-MiniLM-L6-v2`.
   - **Talking Point**:
     > *"TrustDoc AI runs completely locally and offline. All model weights and embedding caches reside on external storage (`E:` drive), keeping inference lightweight and CPU-compatible."*

---

### Step 2: Ingestion & AST Code Graph Tab (~1 min)
1. **Action**: Click the **📁 Sources & Ingestion** tab.
2. **What to point out on screen**:
   - The **Indexed Project Sources** table lists `workers.py` (Python AST-analyzed source) and `READme.md` (Markdown documentation).
   - **Talking Point**:
     > *"TrustDoc AI doesn't just treat code as raw text. During ingestion, it analyzes Python Abstract Syntax Trees (AST) to index code entities—functions, classes, arguments, defaults, and return annotations—with exact line provenance (e.g. `workers.py:5-26`)."*

---

### Step 3: Evidence-Grounded Q&A (`💬 Ask`) (~1.5 min)
1. **Action**: Click the **💬 Ask** tab.
2. **Action**: In the question text area, enter:
   ```text
   What is the mathematical formula used to calculate the retry delay in compute_backoff?
   ```
3. **Action**: Click the primary button **Ask Question**.
4. **What to point out on screen**:
   - **Status Badge**: `ANSWERED` (Green).
   - **Answer Content**:
     > *"In workers.py (lines 5-26), the retry delay in `compute_backoff` is calculated using the formula `base_seconds * 2 ** attempt + random.uniform(0.0, jitter)` [E1]."*
   - **Citations List**: `[E1] workers.py:5-26 (workers.compute_backoff)`.
   - **Evidence Drawer**: Click to expand. Shows the exact retrieved code chunk and line span.
   - **Talking Point**:
     > *"Notice the exact line citation `workers.py:5-26`. The deterministic code-fact resolver derives structural facts directly from the indexed AST code graph in ~15 milliseconds without relying on probabilistic language model generation."*

---

### Step 4: Honest Abstention & False-Premise Refutation (~1.5 min)
Demonstrate how TrustDoc AI honestly refuses to invent facts when evidence is missing.

#### Scenario A: Missing Feature / Out of Scope
1. **Action**: In the **💬 Ask** tab, enter:
   ```text
   What SQLite table stores the execution history of failed tasks?
   ```
2. **Action**: Click **Ask Question**.
3. **What to point out on screen**:
   - **Status Badge**: `INSUFFICIENT_EVIDENCE` (Blue/Info).
   - **Answer**: `The available evidence is insufficient to answer this question.`
   - **Reason**: `below_relevance_threshold` (filtered before prompt construction).
   - **Talking Point**:
     > *"Rather than hallucinating a fictional database schema, the cross-encoder reranker detected that no indexed code supports database persistence and safely abstained."*

#### Scenario B: False-Premise Refutation
1. **Action**: Enter:
   ```text
   Does compute_backoff accept a timeout parameter?
   ```
2. **Action**: Click **Ask Question**.
3. **What to point out on screen**:
   - **Answer**:
     > *"In the indexed source `workers.py` (lines 5-26), `compute_backoff` does not have a `timeout` parameter. Its indexed parameters are: `attempt`, `base_seconds`, `jitter`. Therefore, the question's premise is unsupported by the indexed code [E1]."*
   - **Talking Point**:
     > *"Instead of guessing or hallucinating default values, TrustDoc AI cross-references parameter presence against the AST and refutes false premises with evidence."*

---

### Step 5: Grounded Documentation Generation (`📝 Generate`) (~1.5 min)
1. **Action**: Click the **📝 Generate** tab.
2. **Action**: Select documentation target:
   - Radio button: **Python API Documentation**
   - Entity selectbox: `workers.compute_backoff (function) - line 5`
   - Leave options at default (`run_verify=True`).
3. **Action**: Click **Generate Documentation**.
4. **What to point out on screen**:
   - **Status Badge**: `COMPLETED | VERIFIED` (Green).
   - **Subtab 1 (`📄 Markdown Preview`)**: Shows the generated API documentation containing signature, parameter descriptions (`attempt`, `base_seconds`, `jitter`), return type, and citation tags `[E1]`.
   - **Subtab 2 (`🛡️ Verification & Trust Score`)**:
     - **Trust Score Card**: Displays `100.0 / 100`.
     - **Claim Counts**: `4 supported`, `0 contradicted`, `0 findings`.
   - **Talking Point**:
     > *"Documentation generated by TrustDoc AI is not accepted blindly. It immediately passes through the automated static verification pipeline to ensure every documented parameter and type annotation matches the code."*

---

### Step 6: Automated AST Verification & Trust Score (`🔍 Verify`) (~1.5 min)
Demonstrate the independent verification engine catching documentation bugs and contradictions.

1. **Action**: Click the **🔍 Verify** tab.
2. **Action**: Select radio button: **Verify Custom Markdown Text**.
3. **Action**: Paste deliberately flawed documentation into the text area:
   ```markdown
   The `compute_backoff` function accepts `attempt` and `base_delay`, and returns an integer.

   Call `retry_all()` to requeue everything.
   ```
4. **Action**: Click **Run Verification**.
5. **What to point out on screen**:
   - **Trust Score Card**: Drops to **20.0 / 100**.
   - **Contradiction Penalties Applied**: Shows `-20.0 pts (4 high findings x 5 pts)`.
   - **High Findings Caught**:
     1. `retry_all`: Documented function does not exist in indexed source.
     2. `compute_backoff`: Documented as returning `int`, but actual annotation is `-> float`.
     3. `compute_backoff`: Documented with nonexistent parameter `base_delay` instead of `base_seconds`.
   - **Extracted Claims List**: Expand `📋 Extracted Claims` to show individual claims marked with ❌.
6. **Key Conclusion**:
   > *"The Trust Score is an empirical heuristic combining static AST checks and grounding coverage, not a calibrated mathematical probability of correctness. INSUFFICIENT EVIDENCE indicates an unverified claim, not proof of falsehood."*

---

## 4. Fallback Procedures During Presentation

If any unexpected interruption occurs during the live demonstration:

1. **Streamlit UI Freezes or Port Conflicts**:
   - Restart the UI on an alternative port:
     ```powershell
     .\.venv\Scripts\python.exe -m trustdoc ui --port 8502
     ```
2. **Ollama Connection Drops**:
   - Restart Ollama daemon:
     ```powershell
     $env:OLLAMA_MODELS = "E:\AI\Models\Ollama"
     & "E:\Apps\Ollama\ollama.exe" serve
     ```
3. **Fast Offline CLI Fallback (No Browser Required)**:
   - If a browser issue arises, execute the standalone showcase scripts directly:
     ```powershell
     # Grounded Ask Demo
     .\.venv\Scripts\python.exe examples/ask_demo.py "What is the return type of compute_backoff?"

     # Documentation Verification Demo (Good vs Flawed)
     .\.venv\Scripts\python.exe examples/verify_demo.py
     .\.venv\Scripts\python.exe examples/verify_demo.py --good

     # Grounded Generation Demo
     .\.venv\Scripts\python.exe examples/generate_demo.py --good
     ```

---

## 5. Safe Cleanup Instructions

- **Stop Local Processes**: Simply close the terminal windows running Streamlit and Ollama.
- **Data Preservation**: Do **NOT** delete `data/trustdoc.db` or `data/uploads/`.
- **UI Reset**: Any temporary project created during the demonstration can be safely removed via the **⚙️ Manage Project -> Delete Project** button in the sidebar without affecting the persistent `retry-task-queue` benchmark project.

---

## 6. Methodological Context & Scope

- **Benchmark Scope**: The 20-query canonical dataset and 5-query held-out evaluation set are lightweight benchmark suites specifically designed for rapid, CPU-compatible diagnostic testing.
- **Trust Score Interpretation**: TrustDoc AI enforces rigorous, deterministic AST checks (Tier 1) alongside optional NLI semantic checks (Tier 2). It provides high-confidence protection against documentation drift and API hallucinations while remaining explainable and auditable.
