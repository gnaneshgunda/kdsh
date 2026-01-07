# KDSH 2026 Track A - Updated for Actual Data Format

## Data Structure

### Training CSV (training_data.csv)
```
id | book_name | caption | content | label
46 | In Search × Thalcave | | Thalcave's people faded | consistent
137| The Count × Faria | The Origin of His Connection... | Suspected again in 1815... | consistent
...
```

### Books Folder (./books/)
```
books/
├── In Search.txt          (full narrative)
├── The Count.txt          (full narrative)
├── Kai-Koumou.txt
└── Jacques Paganel.txt
...
```

## How It Works

1. **Load Books**: Read all .txt files from `books/` folder
2. **Load Training Data**: Read CSV with id, book_name, caption, content, label
3. **For Each Row**:
   - Extract book name from "book_name × character"
   - Load corresponding narrative from books/
   - Retrieve relevant sections matching the content snippet
   - Use LLM to reason: is this content consistent?
   - Output prediction (0=Contradict, 1=Consistent)
4. **Save Results**: Write predictions + rationales to results.csv

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env

# Prepare
mkdir -p books
# Copy book .txt files to books/
# Copy training_data.csv to project root

# Run
python rag_system.py --mode batch
```

## Commands

### Batch Process (generates results.csv)
```bash
python rag_system.py --mode batch \
  --books-dir ./books \
  --csv training_data.csv \
  --output results.csv
```

### Interactive (test individual rows)
```bash
python rag_system.py --mode interactive
# Enter: process 46
# Enter: show 137
# Enter: quit
```

## Output Format

results.csv:
```
id | book_name | content | caption | prediction | confidence | rationale
46 | In Search × Thalcave | ... | | 1 | 0.92 | Thalcave's characterization consistent...
137| The Count × Faria | ... | The Origin... | 1 | 0.87 | Historical context supports narrative...
```

## Configuration

### Chunk Size (in rag_system.py, line ~280)
```python
def _chunk_narrative(self, narrative: str, chunk_size: int = 500):
```
- Lower = more chunks, finer retrieval
- Higher = fewer chunks, faster processing

### Retrieval K (line ~310)
```python
top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]  # k=3
```
- Increase to retrieve more context per content snippet

### LLM Model (line ~360)
```python
model="gpt-3.5-turbo"  # Change to gpt-4o-mini for better quality
```

## File Structure

```
project/
├── rag_system.py              # Main system (works with actual format)
├── requirements.txt           # Dependencies
├── .env                       # API keys
├── training_data.csv          # Your training CSV
└── books/                     # Your book .txt files
    ├── In Search.txt
    ├── The Count.txt
    ├── Kai-Koumou.txt
    └── ...
```

## Expected Behavior

### Loading
```
INFO - Loaded book: In Search (45000 chars, 8500 words)
INFO - Loaded book: The Count (52000 chars, 10000 words)
INFO - Loaded training data: 100 rows
Columns: ['id', 'book_name', 'caption', 'content', 'label']
```

### Processing
```
INFO - Processing row 46: In Search × Thalcave...
INFO - Retrieved 3 relevant narrative sections
INFO - Result: 1 (Conf: 0.92)
```

### Output
```
✓ Results written to: results.csv
✓ Total processed: 100
✓ Consistent: 85, Contradict: 15
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No books loaded | Check books/ folder has .txt files |
| CSV not found | Ensure training_data.csv in project root |
| API key error | Create .env with OPENAI_API_KEY |
| Memory error | Reduce chunk_size or process fewer rows |
| No relevant sections | Check narrative content matches CSV topics |

---

**Ready for actual KDSH 2026 data!** 🚀
