import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Assicura il corretto encoding UTF-8 per il terminale Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Configurazione Ambiente e Client Groq
# Cerca il file .env sia nella cartella corrente che nella cartella padre
load_dotenv(override=True)
load_dotenv(dotenv_path="../.env", override=True)

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY non trovata nel file .env!")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)


# 2. Funzione Estrazione Informazioni Strutturate
def estrai_profilo_candidato(testo_cv: str) -> str:
    prompt = f"""Analizza il seguente estratto di curriculum ed estrai le informazioni principali del candidato.
Restituisci ESCLUSIVAMENTE un oggetto JSON valido con la seguente struttura:
{{
    "nome_completo": "Nome e Cognome",
    "titolo_professionale": "Ruolo o specializzazione principale",
    "contatti": {{
        "email": "indirizzo email o Non specificato",
        "telefono": "numero di telefono o Non specificato",
        "citta": "città o Non specificato"
    }}
}}

Estratto CV:
{testo_cv}
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


# 3. Lettura Documenti e Chunking
def carica_documenti(documents_dir: str = "resumes"):
    documents = []
    metadatas = []
    ids = []
    doc_id = 0

    if not os.path.exists(documents_dir):
        print(f"Attenzione: cartella '{documents_dir}' non trovata.")
        return documents, metadatas, ids

    for filename in sorted(os.listdir(documents_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(documents_dir, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read().replace("\n", ". ")
                chunks = [c.strip() for c in content.split("### ") if c.strip()]
                header_summary = chunks[0] if chunks else ""

                for idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "source_file": filename,
                        "header_info": header_summary,
                        "chunk_index": idx
                    })
                    ids.append(f"doc_{doc_id}")
                    doc_id += 1

    print(f"Totale frammenti preparati: {len(documents)}")
    return documents, metadatas, ids


# 4. Pipeline Principale RAG
def main():
    # Caricamento e chunking dei CV
    documents, metadatas, ids = carica_documenti("resumes")
    if not documents:
        print("Nessun documento trovato da indicizzare.")
        return

    # Inizializzazione Embedder con Ollama
    print("Inizializzazione Embedder con Ollama (embeddinggemma-2:270m)...")
    embedder = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name="embeddinggemma-2:270m"
    )

    # Inizializzazione ChromaDB e Popolamento Collezione
    print("Creazione collezione ChromaDB e indicizzazione dei frammenti...")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="talent_pool_cv",
        embedding_function=embedder
    )

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Collezione '{collection.name}' pronta con {collection.count()} documenti.\n")

    # Vector Similarity Search
    job_query = "Cerchiamo un ingegnere software senior con forte esperienza in microservizi, cloud computing e scalabilità ad alte prestazioni"
    print(f"Query di ricerca:\n'{job_query}'\n")

    query_results = collection.query(
        query_texts=[job_query],
        n_results=1
    )

    top_doc = query_results["documents"][0][0]
    top_meta = query_results["metadatas"][0][0]

    # Costruzione Prompt Aumentato (RAG)
    rag_context = f"""--- SCHEDA CANDIDATO INDIVIDUATO ---
File di riferimento: {top_meta.get("source_file")}
Profilo sintetico: {top_meta.get("header_info")}
Paragrafo rilevante del CV:
{top_doc}
------------------------------------"""

    augmented_prompt = f"""Hai a disposizione la seguente scheda informativa estratta dal database dei curricula aziendali:

{rag_context}

Richiesta dell'azienda:
"{job_query}"

Istruzioni per la risposta:
1. Presenta il candidato individuato spiegando con chiarezza perché il suo profilo risponde in modo ottimale alla richiesta.
2. Cita specifiche competenze, tecnologie o risultati concreti menzionati nel testo a supporto della tua raccomandazione.
3. Concludi indicando in modo ordinato i contatti del candidato disponibili per organizzare il colloquio.
"""

    # Generazione Risposta con Groq
    print("Generazione raccomandazione con Groq (openai/gpt-oss-120b)...")
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "Sei un Senior Talent Acquisition Specialist esperto in recruiting tecnico. Formula valutazioni professionali, dettagliate e orientate agli obiettivi aziendali."
            },
            {
                "role": "user",
                "content": augmented_prompt
            }
        ],
        temperature=0.2
    )

    print("\n=== RACCOMANDAZIONE CANDIDATO ===\n")
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
