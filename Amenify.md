# Amenify Summer 2026 – Software Engineering Internship Assignment

Objective:  
This assignment is designed to evaluate your problem-solving ability, coding skills, and engineering judgment. Please complete all sections and submit your work as instructed.

# **Instructions \- READ THESE**

\- Use Python for backend development.  
\- You may use OpenAI APIs, but do not use full existing chatbot frameworks.  
\- Ensure your code is clean, modular, and well-structured.  
\- Include a README with setup steps and explanations.  
\- Submit your code and this document to dchopra@amenify.com.  
\- Include your LinkedIn profile in the submission.

# **Your Task**

Build a basic AI-powered customer support bot for Amenify.

Requirements:  
1\. Use content from amenify.com as the knowledge base.  
2\. Create a Python backend API.  
3\. Build a simple chat UI (HTML/CSS/JavaScript).  
4\. The bot should only answer from the provided knowledge base.  
5\. If the answer is not found, it should respond with 'I don’t know'.  
6\. Maintain chat history within a session.  
7\. Host it somewhere (GCP, AWS, Azure, etc.) and share the link to test the bot.

# **Section 3: Reasoning & Design**

### 1. How did you ingest and structure the data from the website?

**Ingestion:** A two-tier hybrid scraping approach is used in `scraper.py`. First, the system attempts to fetch live content via known sitemap URLs (`sitemap.xml`). If the live site blocks the request (e.g., via Cloudflare 403/503 errors), it gracefully falls back to querying the Wayback Machine CDX API to fetch archived snapshots. BeautifulSoup is used to parse the HTML, aggressively filtering out UI noise (headers, footers, navs, sidebars, scripts) and preferring semantic content containers (`<main>`, `<article>`) over the full document body. Page relevance is validated by checking URL paths and counting brand mentions (`_is_on_topic`).

**Structuring:** The extracted text is processed in `build_kb.py`. It splits text into semantic chunks of roughly 600 characters with a 100-character overlap to preserve context across boundaries. These chunks are embedded locally using the lightweight `sentence-transformers/all-MiniLM-L6-v2` model via the fastembed library (ONNX runtime — no PyTorch required). The normalized embeddings are indexed into a local FAISS vector database (`IndexFlatIP`) using L2-normalized vectors for exact cosine similarity search.

---

### 2. How did you reduce hallucinations?

A multi-layered guardrail strategy is used to strictly confine the LLM to the knowledge base:

**Prompt Engineering & Context Windowing:** The `SYSTEM_PROMPT` explicitly instructs the model to only use the injected `<context>` block and sets a hard rule to reply with "I don't know" if the answer is missing. Temperature is kept low at 0.2.

**Relevance Thresholds:** The retrieval step in `app.py` enforces a strict cosine similarity threshold (0.40, falling back to 0.28). Chunks are also deduplicated per source URL (best 2 per URL, capped at 5 total) to prevent a single noisy page from diluting the context.

**Regex Output Parsing:** `_enforce_idk()` scans the LLM's response for soft uncertain phrasing ("I'm not sure", "I don't have information", "unfortunately I can't"). If detected in a short response (< 160 chars), it overwrites the response with exactly "I don't know", preventing evasive paraphrases.

**Input Sanitization:** A prompt injection detector (`_is_injection`) intercepts attempts to bypass system instructions (e.g., "ignore previous rules", "act as DAN") before they reach the LLM.

**History-Aware Retrieval:** Follow-up questions ("What services does it offer?") are enriched with the previous user turn before vector search, preventing context drift in multi-turn conversations.

---

### 3. What are the limitations of your approach?

**In-Memory State & Free Tier Limits:** The FAISS index and chunk metadata (`chunks.pkl`) are loaded entirely into RAM on startup. On Render's free tier (512MB RAM), scaling the knowledge base to thousands of pages will cause Out-Of-Memory crashes. Currently the KB is 21 pages / 270 chunks, well within limits.

**Blocking Background Tasks:** The knowledge base auto-refresh (`_maybe_refresh_kb`) runs in an `asyncio.to_thread` background task. While non-blocking to the event loop, intensive web scraping and re-embedding on the same single-container instance can spike CPU usage and degrade concurrent API response times.

**Brittle Scraping Logic:** The scraper relies on specific HTML structure and class patterns. A major UI overhaul of amenify.com could break live extraction, forcing reliance on potentially outdated Wayback Machine snapshots.

**Stateless Server / Fat Client:** Chat history is maintained entirely on the client side and passed back in every request payload. Long conversations increase network payload size and consume additional context window tokens on every turn.

---

### 4. How would you scale this system?

**Decouple Storage:** Migrate from local `.index` and `.pkl` files to a managed, distributed vector database (e.g., Pinecone, Qdrant, or Weaviate). This removes the memory bottleneck from the API server entirely.

**Microservice Architecture:** Separate the ingestion/scraping pipeline from the inference API. Scraping and embedding should run on a scheduled background worker queue (e.g., Celery + Redis), updating the vector DB independently without consuming the web server's compute resources.

**Server-Side Session State:** Implement a fast key-value store (Redis) to manage conversation histories by `session_id`. The client would only send the ID and current query, drastically reducing network payload and enabling horizontal scaling across multiple API instances.

---

### 5. What improvements would you make for production use?

**Advanced RAG Pipelines:** Replace the naive query concatenation (`prior_user[-1] + " " + message`) with a dedicated query rewriting step using a smaller LLM to formulate standalone search queries. Add a cross-encoder reranker (e.g., Cohere Rerank) after FAISS retrieval to improve context precision before sending to the LLM.

**Security & Rate Limiting:** Implement IP-based rate limiting via Redis, restrict CORS to the specific frontend origin, and add JWT authentication if the bot is placed behind a user login wall.

**Observability & Telemetry:** Integrate tracing tools (Langfuse or Datadog) to monitor Groq API token usage, latency, retrieval accuracy, and track which sources are most frequently cited in production.

**User Feedback Loop:** Add a thumbs-up/thumbs-down mechanism in the UI to capture user feedback. Store these signals to identify knowledge gaps where the bot frequently triggers the "I don't know" fallback, guiding future scraping and KB expansion.

---

# **Deliverables**

\- Source code  
\- README file  
\- This document with your answers filled in  
\- Example queries and outputs

# **Evaluation Criteria**

\- Code quality and structure  
\- Correctness  
\- Problem-solving approach  
\- Handling of edge cases  
\- Clarity of explanations
