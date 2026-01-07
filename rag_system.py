"""
KDSH 2026 Track A - Narrative Consistency RAG System
Author: Hackathon Participant
Date: January 2026
"""

import os
import json
import csv
import re
import logging
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

client = OpenAI(api_key=API_KEY)

# ---------------------------------------------------------------------
# Data Structure
# ---------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    id: str
    book_char: str
    content: str
    caption: str
    prediction: int
    confidence: float
    rationale: str

# ---------------------------------------------------------------------
# RAG SYSTEM
# ---------------------------------------------------------------------

class KDSHNarrativeConsistencyRAG:

    def __init__(self, books_dir="./books", training_csv="train.csv"):
        self.books_dir = Path(books_dir)
        self.training_csv = Path(training_csv)
        self.narratives: Dict[str, str] = {}

        logger.info("Initialized KDSH Narrative Consistency RAG")
        logger.info(f"Books dir: {self.books_dir}")
        logger.info(f"CSV file: {self.training_csv}")

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------

    def load_books(self) -> Dict[str, str]:
        if not self.books_dir.exists():
            logger.warning("Books directory not found")
            return {}

        for f in self.books_dir.glob("*.txt"):
            key = f.stem.lower()
            self.narratives[key] = f.read_text(encoding="utf-8", errors="ignore")
            logger.info(f"Loaded book: {f.stem}")

        return self.narratives

    def load_training_data(self) -> pd.DataFrame:
        if not self.training_csv.exists():
            raise FileNotFoundError("Training CSV not found")
        df = pd.read_csv(self.training_csv)
        logger.info(f"Loaded {len(df)} training rows")
        return df

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def extract_book_name(self, book_char: str) -> str:
        return book_char.split("×")[0].strip() if "×" in book_char else book_char.strip()

    def chunk_text(self, text: str, size: int = 400) -> List[str]:
        words = text.split()
        return [
            " ".join(words[i:i + size])
            for i in range(0, len(words), size)
            if len(words[i:i + size]) > 50
        ]

    # -----------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------

    def retrieve(self, narrative: str, query: str, k: int = 3):
        chunks = self.chunk_text(narrative)
        q_words = set(query.lower().split())
        scored = []

        for c in chunks:
            overlap = len(q_words & set(c.lower().split()))
            if overlap > 0:
                scored.append((c, overlap / max(len(q_words), 1)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # -----------------------------------------------------------------
    # Reasoning
    # -----------------------------------------------------------------

    def reason(self, book, character, content, caption, narrative):
        evidence = self.retrieve(narrative, content + " " + caption)

        if not evidence:
            return 0, 0.3, "No relevant narrative evidence found"

        evidence_text = "\n\n".join(
            f"Section {i+1}:\n{sec[:300]}..."
            for i, (sec, _) in enumerate(evidence)
        )

        prompt = f"""
You are a narrative consistency judge.

BOOK: {book}
CHARACTER: {character}

CONTENT:
"{content}"

CAPTION:
"{caption}"

NARRATIVE EVIDENCE:
{evidence_text}

Respond ONLY in JSON:
{{
  "consistent": true/false,
  "confidence": 0.5-1.0,
  "reasoning": "short explanation"
}}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=250,
                timeout=30
            )

            text = response.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)

            if not match:
                return 0, 0.5, "Malformed model output"

            data = json.loads(match.group())
            return (
                1 if data.get("consistent") else 0,
                float(data.get("confidence", 0.5)),
                data.get("reasoning", "")
            )

        except Exception as e:
            logger.error(f"LLM failure: {e}")
            return 0, 0.25, "LLM error / timeout"

    # -----------------------------------------------------------------
    # Row Processing
    # -----------------------------------------------------------------

    def process_row(self, row: pd.Series) -> ConsistencyResult:
        book_char = str(row["book_name"])
        content = str(row["content"])
        caption = str(row.get("caption", ""))

        book = self.extract_book_name(book_char)
        narrative = self.narratives.get(book.lower(), "")

        if not narrative:
            return ConsistencyResult(
                id=str(row["id"]),
                book_char=book_char,
                content=content,
                caption=caption,
                prediction=0,
                confidence=0.2,
                rationale="Narrative not found"
            )

        pred, conf, reason = self.reason(
            book, book_char, content, caption, narrative
        )

        return ConsistencyResult(
            id=str(row["id"]),
            book_char=book_char,
            content=content,
            caption=caption,
            prediction=pred,
            confidence=conf,
            rationale=reason
        )

    # -----------------------------------------------------------------
    # Batch Mode (INCREMENTAL CSV)
    # -----------------------------------------------------------------

    def batch_process(self, output_csv="results.csv"):
        self.load_books()
        df = self.load_training_data()

        output_path = Path(output_csv)
        file_exists = output_path.exists()

        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "id",
                    "book_name",
                    "content",
                    "caption",
                    "prediction",
                    "confidence",
                    "rationale"
                ])
                f.flush()

            for _, row in df.iterrows():
                row_id = row["id"]
                logger.info(f"Processing row {row_id}")

                try:
                    result = self.process_row(row)

                    writer.writerow([
                        result.id,
                        result.book_char,
                        result.content[:200],
                        result.caption[:200],
                        result.prediction,
                        f"{result.confidence:.2f}",
                        result.rationale[:200]
                    ])
                    f.flush()

                    logger.info(
                        f"Saved row {row_id} | pred={result.prediction} | conf={result.confidence:.2f}"
                    )

                except Exception as e:
                    logger.error(f"Failed row {row_id}: {e}")

                time.sleep(1.2)  # rate-limit safe

        logger.info("Batch processing complete")

    # -----------------------------------------------------------------
    # Interactive Mode
    # -----------------------------------------------------------------

    def interactive(self):
        self.load_books()
        df = self.load_training_data()

        print("\nInteractive Mode (process <id> | quit)\n")

        while True:
            cmd = input(">> ").strip()
            if cmd == "quit":
                break

            if cmd.startswith("process"):
                rid = int(cmd.split()[1])
                row = df[df["id"] == rid].iloc[0]
                res = self.process_row(row)

                print("\nRESULT")
                print("Prediction:", "CONSISTENT" if res.prediction else "CONTRADICT")
                print("Confidence:", res.confidence)
                print("Reason:", res.rationale)
                print()

# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "interactive"], default="batch")
    parser.add_argument("--books-dir", default="./books")
    parser.add_argument("--csv", default="train.csv")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    rag = KDSHNarrativeConsistencyRAG(args.books_dir, args.csv)

    if args.mode == "batch":
        rag.batch_process(args.output)
    else:
        rag.interactive()

if __name__ == "__main__":
    main()
