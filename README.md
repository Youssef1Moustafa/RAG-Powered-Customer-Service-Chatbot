\# 📞 Telecom Egypt Intelligent Assistant (WE Chatbot)



> RAG-powered chatbot trained on the official Telecom Egypt website (te.eg), supporting Arabic, English, and Egyptian dialect.



---



\## 🧠 System Overview



This system is a \*\*Retrieval-Augmented Generation (RAG)\*\* chatbot that answers customer queries using the Telecom Egypt official website as its knowledge base. It runs \*\*100% locally\*\* — no cloud, no API keys required.



```

User Question

&#x20;    │

&#x20;    ▼

&#x20;Embed Question (nomic-embed-text via Ollama)

&#x20;    │

&#x20;    ▼

&#x20;ChromaDB Vector Search (MMR, top-6)

&#x20;    │

&#x20;    ▼

&#x20;Retrieved Chunks + Prompt

&#x20;    │

&#x20;    ▼

&#x20;LLM (llama3.2:3b via Ollama)

&#x20;    │

&#x20;    ▼

&#x20;Answer + Source Citations

```



\---



\## 📁 Project Structure



```

project/

│

├── app.py                  ← Streamlit chat interface

├── rag\_pipeline.py         ← RAG engine (embeddings, retrieval, generation)

├── document\_processor.py   ← File upload handler (PDF, DOCX, TXT, Images, HTML)

├── scraper.py              ← Web scraper for te.eg

│

├── requirements.txt        ← Python dependencies

├── README.md               ← This file

│

├── data/

│   ├── website\_pages/      ← Scraped pages (auto-created by scraper.py)

│   └── chroma\_db/          ← Vector database (auto-created on first load)

```



\---



\## ⚙️ Prerequisites



| Requirement | Version | Notes |

|-------------|---------|-------|

| Python | 3.10+ | |

| Ollama | Latest | https://ollama.com/download |

| Tesseract OCR | 5.x | For image files (optional) |



\### Install Tesseract (Windows)

Download from: https://github.com/UB-Mannheim/tesseract/wiki  

Install to: `C:\\Program Files\\Tesseract-OCR\\`  

Add Arabic language pack during installation.



\---



\## 🚀 Setup \& Installation



\### Step 1 — Clone / Download the project

```bash

cd C:\\your\\path\\

```



\### Step 2 — Create virtual environment

```bash

python -m venv venv

venv\\Scripts\\activate

```



\### Step 3 — Install dependencies

```bash

pip install --upgrade pip

pip install -r requirements.txt

```



\### Step 4 — Install Ollama models

```bash

\# Download Ollama from https://ollama.com/download then run:

ollama pull llama3.2:3b       # Main LLM (\~2GB)

ollama pull nomic-embed-text  # Embeddings model (\~274MB)

```



\### Step 5 — Scrape the website (first time only)

```bash

python scraper.py

```

This creates `data/website\_pages/` with \~40 pages from te.eg.



\---



\## ▶️ Running the App



\*\*Terminal 1 — Start Ollama server:\*\*

```bash

ollama serve

```



\*\*Terminal 2 — Start the chatbot:\*\*

```bash

streamlit run app.py

```



Open your browser at: \*\*http://localhost:8501\*\*



\---



\## 💬 How to Use



1\. Click \*\*"تحميل البيانات"\*\* in the sidebar to build the vector database

2\. Wait for the success message (takes \~2-3 minutes first time)

3\. Start asking questions in Arabic or English

4\. Optionally upload PDF/DOCX/TXT/Image files for additional context



\### Example Questions

\- `ما هي باقات الإنترنت المتاحة؟`

\- `How do I pay my WE bill?`

\- `إيه الفرق بين Prepaid و Postpaid؟`

\- `What is the WE Gold number service?`



\---



\## 📂 Supported File Formats



| Format | Support |

|--------|---------|

| PDF | ✅ Text extraction |

| DOCX | ✅ Full support |

| TXT | ✅ UTF-8 + Arabic (cp1256) |

| PNG / JPG / JPEG | ✅ OCR (requires Tesseract) |

| HTML / HTM | ✅ BeautifulSoup parsing |



\---



\## 🔧 Configuration



In `rag\_pipeline.py` you can change:



```python

\# LLM Model

TelecomRAG(model\_name="llama3.2:3b")   # default

TelecomRAG(model\_name="llama3.2:1b")   # faster, less accurate

TelecomRAG(model\_name="mistral")        # alternative



\# Retrieval settings

search\_type="mmr"           # MMR reduces redundancy

search\_kwargs={"k": 6}      # Number of chunks retrieved

```



\---



\## 🏗️ System Architecture



| Component | Technology |

|-----------|-----------|

| Web Scraping | requests + BeautifulSoup |

| Text Chunking | LangChain RecursiveCharacterTextSplitter |

| Embeddings | nomic-embed-text (Ollama) |

| Vector Store | ChromaDB (local) |

| LLM | llama3.2:3b (Ollama) |

| RAG Framework | LangChain RetrievalQA |

| Interface | Streamlit |

| Language Detection | langdetect |



\---



\## ❗ Troubleshooting



| Problem | Solution |

|---------|----------|

| `ollama: command not found` | Restart terminal after installing Ollama |

| `Connection refused` on port 11434 | Run `ollama serve` in a separate terminal |

| No answer / empty response | Click "تحميل البيانات" or "إعادة تحميل" |

| Image OCR not working | Install Tesseract + Arabic language pack |

| Slow responses | Normal for first query — model loads into RAM |



\---



\## 📋 Requirements



```

streamlit

langchain

langchain-community

langchain-ollama

langchain-text-splitters

chromadb

pypdf

python-docx

pillow

pytesseract

langdetect

beautifulsoup4

requests

lxml

```



\---



\*Built as a Proof of Concept for Telecom Egypt customer service automation.\*  

\*All data sourced from the official website: https://te.eg/wps/portal/te/Personal*

