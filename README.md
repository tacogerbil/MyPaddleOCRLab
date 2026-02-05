# Home Lab OCR + LLM Cleanup Workflow

This document describes a home lab setup for converting scanned books (image-only PDFs) into clean, readable text using **PaddleOCR** and a **local LLM** (Llama 3.1 8B) for corrections. No summarization is performed — only error correction and paragraph normalization.

---

## 1. Hardware Requirements

- **CPU:** Intel Core i7 or equivalent
- **GPU:** NVIDIA RTX 5070 Ti (16 GB VRAM) for LLM inference
- **Memory:** 16 GB+ recommended
- **Disk:** Enough to store scanned PDFs, OCR output, and cleaned pages (HDD preferred for large libraries)
- **OS:** Linux-based (PopOS in this setup)

---

## 2. Software Components

| Component | Purpose |
|-----------|--------|
| PaddleOCR (Docker) | Extract text from scanned images/PDFs |
| Local LLM (Llama 3.1 8B, quantized) | Correct OCR mistakes, normalize paragraphs |
| n8n (Workflow Orchestrator) | Automate processing pipeline |
| Optional post-processing scripts | Merge pages into final book, minor cleanup if needed |

---

## 3. Workflow Overview

```
Book PDF (images only)
       ↓
PaddleOCR → OCR JSON/text
       ↓
Page-by-page → Local LLM (Llama 3.1 8B)
       ↓
Cleaned Text (paragraphs normalized, OCR errors fixed)
       ↓
Optional: merge all pages → final book text
```

### Notes:
- Each page is processed individually to stay within LLM context limits.
- No summarization is performed.
- LLM handles corrections intelligently, avoiding blind pattern-based mistakes.

---

## 4. LLM Role

- Correct misrecognized characters (`0` → `O`, `1` → `l`, etc.)
- Merge broken lines into coherent sentences
- Fix hyphenation (`exam-\nple` → `example`)
- Normalize paragraphs without losing meaning

**Prompt example (for HTTP API wrapper or CLI input):**
```
Input: {{ OCR page text }}
Task: Correct OCR errors, fix line breaks and hyphenation, normalize paragraphs.
Output: Cleaned text preserving original meaning.
```

---

## 5. n8n Workflow Blueprint

### Nodes:

1. **Trigger Node**
   - Type: `Watch Folder`
   - Folder: `/media/HDD/ocr/data`
   - Fires when new PDFs/images arrive

2. **PaddleOCR Node**
   - Type: `Execute Command` or Docker API call
   - Runs PaddleOCR container on new file
   - Mounts:
     - Input: `/media/HDD/ocr/data`
     - Output: `/media/HDD/ocr/output`

3. **Page Split / Iterate Node**
   - Splits large OCR output into pages or 500–1000 token chunks
   - Passes each chunk to LLM

4. **LLM Cleanup Node**
   - Type: `HTTP Request` or `Execute Command` (for local LLM)
   - Performs context-aware correction and paragraph normalization

5. **Save Output Node**
   - Saves cleaned page as `.txt` in `/media/HDD/ocr/cleaned_pages`

6. **Optional Merge Node**
   - Combines all cleaned pages into a single text file
   - Example command: `cat /media/HDD/ocr/cleaned_pages/*.txt > /media/HDD/ocr/final_book.txt`

### Folder Structure:
```
/media/HDD/ocr/data          # Input scanned PDFs/images
/media/HDD/ocr/output        # PaddleOCR raw text
/media/HDD/ocr/cleaned_pages # LLM-corrected pages
/media/HDD/ocr/final_book.txt # Optional merged final book
```

---

## 6. Best Practices

- Process **one page per LLM session** to avoid context overflow.
- Keep original OCR JSON for reference or debugging.
- Optional: use lightweight post-processing scripts for minor merges or checks.
- LLM-only cleanup step ensures **context-aware correction**, preserving meaning.

---

## 7. Model Recommendations

| Model | VRAM Fit (RTX 5070 Ti) | Strengths |
|-------|------------------------|----------|
| Llama 3.1 8B (quantized) | Very good | General-purpose, context-aware text cleanup, efficient VRAM usage |
| Qwen2.5-VL 7B | Good | Lightweight, instruction-following, acceptable for simpler cleanup tasks |
| DeepSeek-R1-Distill-Qwen-14B-Q4_K_L | Tight fit | Structured/retrieval-focused, less ideal for messy OCR cleanup |
| Qwen2.5-14B-Instruct-Q4_K_M | Very tight | Instruction-following, VRAM heavy, slower on local GPU |

**Recommendation:** Use **Llama 3.1 8B** for OCR correction and paragraph normalization. Other models may be used experimentally but are less ideal for your use case.

---

## 8. Summary

This workflow allows you to:
- Convert scanned book PDFs into clean, readable text
- Correct OCR mistakes intelligently
- Normalize paragraphs without losing meaning
- Automate the process with n8n
- Run entirely on local hardware (RTX 5070 Ti, 16 GB VRAM)

No summarization or semantic interpretation is done — purely **correction and cleanup**.

---

End of document.


