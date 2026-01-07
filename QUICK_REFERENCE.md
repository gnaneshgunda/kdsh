# KDSH 2026 - Quick Reference

## All Commands (Copy-Paste Ready)

### Setup (5 min)
```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
mkdir -p books && cp /your/books/*.txt books/
```

### Prepare Data
```bash
# Export Excel to CSV: training_data.csv
# Copy book .txt files to books/ folder

# Verify
ls -la training_data.csv
ls -la books/*.txt
```

### Run
```bash
# Batch (process all rows)
python rag_system.py --mode batch

# Interactive (test single rows)
python rag_system.py --mode interactive
```

### Check Results
```bash
# View predictions
cat results.csv

# Summary
echo "Consistent:" && grep ",1," results.csv | wc -l
echo "Contradict:" && grep ",0," results.csv | wc -l
```

## Input Format

### training_data.csv
```
id,book_name,caption,content,label
46,"In Search × Thalcave","","Thalcave's people faded",consistent
137,"The Count × Faria","The Origin of His Connection...","Suspected again in 1815...",consistent
```

### books/ folder
```
books/
├── In Search.txt
├── The Count.txt
├── Kai-Koumou.txt
└── Jacques Paganel.txt
```

## Output Format

### results.csv
```
id,book_name,content,caption,prediction,confidence,rationale
46,In Search × Thalcave,Thalcave's people faded,,1,0.92,Character inconsistency...
137,The Count × Faria,Suspected again in 1815...,The Origin of...,1,0.87,Timeline suggests...
```

## Common Issues

| Problem | Fix |
|---------|-----|
| FileNotFoundError: training_data.csv | Export spreadsheet as CSV in project root |
| No books loaded | Check books/ folder, verify .txt files exist |
| OPENAI_API_KEY error | Create .env with valid key: `echo "OPENAI_API_KEY=sk-..." > .env` |
| ModuleNotFoundError | Run: `pip install -r requirements.txt` |
| API rate limit | Wait 60s or use gpt-3.5-turbo (cheaper) |

## Key Parameters (rag_system.py)

- **chunk_size** (line ~281): Size of narrative chunks (500 words default)
- **k** (line ~314): Number of relevant sections retrieved (3 default)
- **model** (line ~360): LLM to use (gpt-3.5-turbo default)
- **temperature** (line ~363): 0.3 (low=consistent, high=creative)

---

**Status: Ready for actual KDSH 2026 data ✓**
