# Selfwork: RAG Base (Retrieval-Augmented Generation)

Implementazione di una pipeline **RAG Base** in Python per l'ingestione, l'indicizzazione vettoriale e la ricerca semantica di profili professionali (Curricula) utilizzando:
- **ChromaDB**: Vector database in-memory per l'indicizzazione dei documenti e similarity search.
- **Ollama (`embeddinggemma-2:270m`)**: Embedder per generare rappresentazioni vettoriali a 768 dimensioni (con fallback automatico su `DefaultEmbeddingFunction`).
- **Groq / OpenAI**: LLM per l'estrazione di informazioni strutturate e la generazione della raccomandazione finale arricchita con il contesto recuperato.

---

## 📁 Struttura del Progetto

```
Selfwork-Rag-Base/
├── .env.example              # Template variabili d'ambiente
├── .gitignore                # File e cartelle ignorati da Git
├── README.md                 # Documentazione del progetto
├── rag_base.py               # Script Python principale con il flusso RAG
├── requirements.txt          # Dipendenze Python
└── resumes/                  # Cartella contenente i curricula (.txt)
    ├── curriculum-controllo-di-gestione.txt
    ├── curriculum-cybersecurity.txt
    ├── curriculum-marketing.txt
    └── curriculum-software-engineer.txt
```

---

## 🚀 Installazione e Setup

### 1. Attiva il virtual environment
```bash
# Da Git Bash (se usi il venv del workspace)
source ../.venv/Scripts/activate

# Oppure da PowerShell
..\.venv\Scripts\Activate.ps1
```

### 2. Installa le dipendenze
```bash
pip install -r requirements.txt
```

### 3. Configura le variabili d'ambiente
Copia il file `.env.example` in `.env` (se non già presente nella root del workspace) e inserisci la tua API Key:
```env
GROQ_API_KEY=gsk_la_tua_chiave_groq
```

### 4. (Opzionale) Modello Ollama per Embeddings
Se desideri utilizzare gli embedding di Ollama:
```bash
ollama pull embeddinggemma-2:270m
```
*(Nota: se Ollama non è in esecuzione, lo script utilizzerà automaticamente l'embedder locale predefinito di ChromaDB).*

---

## ▶️ Esecuzione dello Script

Per avviare la pipeline completa (chunking, indicizzazione, query semantica e risposta aumentata):
```bash
python rag_base.py
```

