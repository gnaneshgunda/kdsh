# Setup for KDSH 2026 with Your Data

## Prerequisites

- Python 3.9+
- OpenAI API key
- training_data.csv (your training spreadsheet as CSV)
- books/ folder with .txt narrative files

## Step 1: Export Your Spreadsheet to CSV

In Excel/Google Sheets:
1. Open your training_data.xlsx
2. File → Export as CSV → training_data.csv
3. Save to project root

Expected columns:
- id
- book_name (or "book_char")
- caption (optional)
- content
- label (for reference, not used)

## Step 2: Organize Books Folder

```bash
mkdir -p books

# Copy all your book .txt files
cp /path/to/your/books/*.txt books/
```

Naming should match book_name in CSV:
- CSV: "In Search × Thalcave" → File: "In Search.txt"
- CSV: "The Count × Faria" → File: "The Count.txt"
- etc.

## Step 3: Setup Python Environment

```bash
# Create venv
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate
# Or (Windows)
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Configure API Key

```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-actual-key" > .env

# Verify
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Key set:', bool(os.getenv('OPENAI_API_KEY')))"
```

## Step 5: Run

```bash
# Batch process all rows
python rag_system.py --mode batch

# Or interactive for testing
python rag_system.py --mode interactive
```

## Step 6: Check Results

```bash
# View generated results
cat results.csv

# Count predictions
grep ",1," results.csv | wc -l    # Consistent count
grep ",0," results.csv | wc -l    # Contradict count
```

## Data Format Mapping

Your spreadsheet columns → CSV columns:

| Spreadsheet | CSV Header | Example |
|-------------|-----------|---------|
| id | id | 46 |
| book_name|char | book_name | In Search × Thalcave |
| (empty) | caption | The Origin of His Connection... |
| content | content | Thalcave's people faded |
| consistent/contradict | label | consistent |

## Expected Files After Setup

```
project/
├── rag_system.py
├── requirements.txt
├── training_data.csv              ← Your exported CSV
├── .env                           ← Your API key
├── books/
│   ├── In Search.txt              ← Your book files
│   ├── The Count.txt
│   ├── Kai-Koumou.txt
│   └── ...
└── results.csv                    ← Generated output
```

## Verification Checklist

```bash
# Check files exist
ls -la training_data.csv
ls -la .env
ls -la books/*.txt

# Check CSV format
head -n 3 training_data.csv

# Check books loaded
python -c "
from pathlib import Path
books = list(Path('books').glob('*.txt'))
print(f'Found {len(books)} book files')
for b in books: print(f'  - {b.name}')
"

# Verify API key
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('API Key configured:', bool(os.getenv('OPENAI_API_KEY')))
"
```

---

**Now ready to process your actual training data!**
