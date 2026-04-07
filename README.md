# Week 3: Pretraining Data Collection & Voice Agents

**Course:** Machine Learning Engineer in the Generative AI Era
**Week:** 3 of 10
**Topics:** Web Scraping, OCR, ASR, Data Cleaning, TTS, Voice Agents

---

## Overview

This week you'll build a complete pretraining data pipeline -- from web scraping and OCR to data cleaning and deduplication -- mirroring what companies like Meta (LLaMA 4) and HuggingFace (FineWeb) use to prepare training data for frontier LLMs. You'll also build a voice agent that combines speech recognition, LLM reasoning, and text-to-speech into a conversational pipeline.

The homework is updated for 2025-2026 with modern tools: Crawl4AI for LLM-ready web extraction, Marker/Docling for PDF processing, faster-whisper for ASR, Kokoro for TTS, and Pipecat for voice agents.

---

## Learning Objectives

1. Extract clean text from web pages using trafilatura and Crawl4AI
2. Perform OCR on images and PDFs with Tesseract, EasyOCR, Marker, and Docling
3. Transcribe audio using faster-whisper with model size trade-offs
4. Build a data cleaning pipeline with MinHash deduplication and PII removal
5. Synthesize speech with edge-tts and Kokoro TTS
6. Design a voice agent pipeline (ASR -> LLM -> TTS)

---

## Setup Options

### Path A: Claude API (Cloud) -- Recommended

- **Default model:** `claude-sonnet-4-6`
- **Cost:** ~$1-3 for the full assignment
- **Requires:** `ANTHROPIC_API_KEY` in `.env`

### Path B: Ollama (Local / Free)

- **Default model:** `qwen3.5:27b`
- **Requires:** ~20GB RAM, `ollama pull qwen3.5:27b`

### Path C: Hybrid

- Use both Claude + Ollama

---

## Prerequisites

- Python 3.11+
- ffmpeg (for audio processing)
- tesseract (for OCR)

### System Dependencies

```bash
# macOS
brew install tesseract ffmpeg

# Ubuntu/Debian
sudo apt install tesseract-ocr ffmpeg
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Homework3-Submission

# 2. Create virtual environment (recommended)
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 3. Install Python packages
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 5. (Path B/C only) Start Ollama
ollama serve
ollama pull qwen3.5:27b
```

---

## Repository Structure

```
Homework3-Submission/
├── README.md
├── requirements.txt
���── .env.example
├── .gitignore
├── notebooks/
│   ├── 00_setup_verification.ipynb       # 5 min
��   ├── 01_environment_setup.ipynb        # 20 min
│   ├── 02_web_scraping.ipynb             # 30 min
│   ├── 03_document_ocr.ipynb             # 30 min
│   ├─�� 04_speech_recognition.ipynb       # 25 min
│   ├── 05_data_cleaning.ipynb            # 30 min
│   ├── 06_text_to_speech.ipynb           # 25 min
│   ├── 07_voice_agent.ipynb              # 30 min
│   └── 08_project_integration.ipynb      # 35 min
├── src/
│   ├── __init__.py
���   ├── config.py                         # PATH, model settings
│   ├── llm_client.py                     # Claude + Ollama unified client
│   ├── cost_tracker.py                   # API cost tracking
│   ├── utils.py                          # Formatting, reflection helpers
│   ├── prompt_templates.py               # CO-STAR framework
│   ├── scraping_utils.py                 # Web scraping (trafilatura, Crawl4AI)
│   ├── ocr_utils.py                      # OCR (Tesseract, EasyOCR, Marker, Docling)
│   ├── audio_utils.py                    # ASR (faster-whisper) + TTS (Kokoro, edge-tts)
│   └── data_pipeline.py                  # Data cleaning (dedup, PII, quality)
├── outputs/                              # Auto-created by notebooks
│   ├── homework_reflection.md            # Primary deliverable
│   ├── my_project_update.md              # Project integration
│   ├── arxiv_papers.json                 # Scraped papers
│   └── cleaned_data.json                 # Pipeline output
├── test_data/                            # Sample data for experiments
│   ├── audio/
│   ├── image/
│   └── data/
└── docs/
```

---

## Assignment Structure

| Notebook | Topic | Time | Key Deliverable |
|----------|-------|------|-----------------|
| 00 | Setup Verification | 5 min | -- |
| 01 | Environment Setup | 20 min | path_selection.md |
| 02 | Web Scraping & Text Extraction | 30 min | arxiv_papers.json |
| 03 | Document OCR & PDF Extraction | 30 min | OCR comparison analysis |
| 04 | Speech Recognition (ASR) | 25 min | Transcription comparison |
| 05 | Data Cleaning Pipeline | 30 min | cleaned_data.json |
| 06 | Text-to-Speech & Voice Synthesis | 25 min | Audio files |
| 07 | Voice Agent with Pipecat | 30 min | Multi-turn conversation |
| 08 | Project Integration | 35 min | my_project_update.md |

**Total estimated time:** ~4 hours

---

## Deliverables

1. `outputs/homework_reflection.md` (70%) -- Built incrementally across all notebooks
2. `outputs/my_project_update.md` (20%) -- Data pipeline strategy for your capstone
3. All notebooks executed with TODOs filled in (10%)

---

## Cost Estimates

| Path | Model | Estimated Cost |
|------|-------|---------------|
| A | claude-sonnet-4-6 | ~$1.50-3.00 |
| B | qwen3.5:27b (Ollama) | Free |
| C | Hybrid | ~$0.50-1.50 |

---

## Troubleshooting

**Tesseract not found:**
```bash
brew install tesseract  # macOS
sudo apt install tesseract-ocr  # Linux
```

**ffmpeg not found:**
```bash
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Linux
```

**Ollama connection refused:**
```bash
ollama serve  # Start the server
ollama pull qwen3.5:27b  # Pull the default model
```

**edge-tts fails in Jupyter:**
```bash
pip install nest-asyncio
# Then add at top of notebook: import nest_asyncio; nest_asyncio.apply()
```

**Marker/Docling import errors:**
These are optional packages that require PyTorch. Install separately:
```bash
pip install marker-pdf  # For Marker
pip install docling      # For Docling
```

---

## Bonus Challenges (Optional)

1. **Crawl4AI deep dive** (+5%): Install Crawl4AI and compare with trafilatura on 10+ URLs
2. **Kokoro TTS** (+5%): Install Kokoro and compare audio quality with edge-tts
3. **Full voice pipeline** (+10%): Build a working FastAPI voice server with ASR->LLM->TTS
4. **Pipecat integration** (+10%): Build a Pipecat voice agent with WebRTC transport

---

## Resources

### Web Scraping
- [trafilatura docs](https://trafilatura.readthedocs.io/)
- [Crawl4AI docs](https://docs.crawl4ai.com/)

### OCR & Document Extraction
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Marker PDF](https://github.com/VikParuchuri/marker)
- [Docling (IBM)](https://github.com/DS4SD/docling)

### ASR & TTS
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Kokoro TTS](https://github.com/hexgrad/kokoro)
- [edge-tts](https://github.com/rany2/edge-tts)

### Data Cleaning
- [DataTrove (HuggingFace)](https://github.com/huggingface/datatrove)
- [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [datasketch (MinHash)](https://ekzhu.com/datasketch/)

### Voice Agents
- [Pipecat](https://github.com/pipecat-ai/pipecat)
- [Meta LLaMA 4 Blog](https://ai.meta.com/blog/meta-llama-4/)

---

## Timeline (7-Day Schedule)

| Day | Notebooks | Focus |
|-----|-----------|-------|
| 1 | 00, 01 | Setup and path selection |
| 2 | 02 | Web scraping and text extraction |
| 3 | 03 | OCR and document extraction |
| 4 | 04, 05 | ASR and data cleaning pipeline |
| 5 | 06 | TTS and voice synthesis |
| 6 | 07 | Voice agent pipeline |
| 7 | 08 | Project integration and submission |

---

## Support

- Discord: #week-3 channel
- Office Hours: Check course calendar
