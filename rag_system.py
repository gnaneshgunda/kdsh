"""
KDSH 2026 Track A – Narrative Consistency RAG (NVIDIA NIM Edition)
Transitioned from OpenAI to NVIDIA AI Foundation Endpoints.
"""

import os
import json
import csv
import time
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Use the NVIDIA LangChain integration or standard OpenAI-compatible client
from openai import OpenAI

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# NVIDIA NIMs are OpenAI-API compatible! 
# You just change the base_url and use your NVIDIA API Key.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

@dataclass
class Chunk:
    text: str
    embedding: np.ndarray
    kind: str 

# --- RAG Core ---
class NarrativeConsistencyRAG:
    def __init__(self, books_dir="./books", csv_path="train.csv", index_path="nvidia_narrative_index.pkl"):
        self.books_dir = Path(books_dir)
        self.csv_path = Path(csv_path)
        self.index_path = Path(index_path)
        self.corpus: Dict[str, List[Chunk]] = {}

    def embed(self, texts: List[str]) -> np.ndarray:
        """Fetch embeddings from NVIDIA's NV-Embed-QA model."""
        # Using NVIDIA's retrieval-optimized embedding model
        response = client.embeddings.create(
            input=texts,
            model="nvidia/nv-embedqa-e5-v5", # One of NVIDIA's best embedding models
            extra_body={"input_type": "query"}
        )
        return np.array([e.embedding for e in response.data])

    def chunk_text(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        chunks, buf = [], []
        for s in sentences:
            buf.append(s)
            if len(" ".join(buf).split()) > 180:
                chunks.append(" ".join(buf))
                buf = []
        if buf: chunks.append(" ".join(buf))
        return chunks

    def classify_kind(self, chunk: str) -> str:
        triggers = ["never", "always", "habit", "trait", "hated", "fears", "childhood", "forbidden"]
        return "backstory" if any(t in chunk.lower() for t in triggers) else "event"

    def build_or_load(self):
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                self.corpus = pickle.load(f)
            logger.info("Loaded cached NVIDIA index.")
        else:
            logger.info("Building NVIDIA index...")
            for f in self.books_dir.glob("*.txt"):
                book_key = f.stem.lower()
                text = f.read_text(encoding="utf-8", errors="ignore")
                txt_chunks = self.chunk_text(text)
                
                # NVIDIA NIMs usually have high rate limits, 
                # but batching is still better.
                embeddings = self.embed(txt_chunks)
                kinds = [self.classify_kind(c) for c in txt_chunks]
                
                self.corpus[book_key] = [
                    Chunk(text=c, embedding=e, kind=k)
                    for c, e, k in zip(txt_chunks, embeddings, kinds)
                ]
                logger.info(f"Indexed: {book_key}")
            
            with open(self.index_path, "wb") as f:
                pickle.dump(self.corpus, f)

    def retrieve_and_rerank(self, book_key: str, character: str, query: str, k=3):
        search_query = f"Retrieve narrative evidence for: {character} - {query}"
        q_emb = self.embed([search_query])[0]
        
        chunks = self.corpus.get(book_key, [])
        if not chunks: return [], [], True

        emb_matrix = np.stack([c.embedding for c in chunks])
        sims = cosine_similarity([q_emb], emb_matrix)[0]
        
        top_indices = np.argsort(sims)[-15:][::-1]
        candidates = [chunks[i] for i in top_indices]
        candidate_sims = [sims[i] for i in top_indices]

        events = [c.text for c, s in zip(candidates, candidate_sims) if c.kind == "event" and s > 0.22][:k]
        backstory = [c.text for c, s in zip(candidates, candidate_sims) if c.kind == "backstory" and s > 0.18][:k]
        
        return events, backstory, (max(sims) < 0.25)

    def judge(self, book, content, caption, events, backstory, absence):
        prompt = f"Book: {book}\nStatement: {content}\nEvents: {events}\nBackstory: {backstory}\nAbsence: {absence}"
        
        # Using Llama-3-70b-Instruct via NVIDIA NIM
        try:
            r = client.chat.completions.create(
                model="meta/llama3-70b-instruct", 
                messages=[{"role": "system", "content": "Return ONLY JSON with 'consistent' (bool), 'confidence' (float), and 'reasoning' (str)."},
                          {"role": "user", "content": prompt}],
                temperature=0.1
            )
            res = json.loads(r.choices[0].message.content)
            return (1 if res["consistent"] else 0, float(res["confidence"]), res["reasoning"])
        except Exception as e:
            logger.error(f"Judging Error: {e}")
            return (0, 0.5, "NVIDIA NIM processing error")

    def run_pipeline(self, output_file="results.csv"):
        self.build_or_load()
        df = pd.read_csv(self.csv_path)
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prediction", "confidence", "rationale"])
            for _, row in df.iterrows():
                parts = row["book_name"].split("×")
                book_key, char_name = parts[0].strip().lower(), (parts[1].strip() if len(parts) > 1 else "Character")
                ev, bk, ab = self.retrieve_and_rerank(book_key, char_name, str(row["content"]))
                pred, conf, reason = self.judge(book_key, row["content"], row.get("caption", ""), ev, bk, ab)
                writer.writerow([row["id"], pred, f"{conf:.2f}", reason])
                f.flush()

if __name__ == "__main__":
    rag = NarrativeConsistencyRAG()
    rag.run_pipeline()