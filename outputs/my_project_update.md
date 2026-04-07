# Week 3 Project Update: Pretraining Data & Voice Agents

## Project Description

[STUDENT: Describe your capstone project in 2-3 sentences]

## Data Pipeline Strategy

# Data Pipeline Strategy — Week 2 Capstone Review

---

## ⚠️ Important Notice Before We Begin

Your project definition fields are **all still placeholder text**. I can see:
- Project description: `[STUDENT: Describe your capstone project...]`
- Domain: `[STUDENT: e.g., healthcare, finance...]`
- Problem/Solution/Components: All unfilled template text

**I cannot give you specific, useful pipeline advice without knowing what you're actually building.**

Giving you generic advice here would waste your time and produce a pipeline that may be completely wrong for your use case.

---

## What You Need To Do Right Now

**Fill in your project definition with real content.** Here's a concrete example of what good vs. bad input looks like:

| ❌ What you submitted | ✅ What I need |
|---|---|
| `[Your project name]` | "Legal Brief Summarization Agent" |
| `[What research problem are you solving?]` | "Law students can't quickly extract key arguments from 200-page case files" |
| `[How will your agent solve this?]` | "RAG pipeline over uploaded PDFs with citation-aware summarization" |

---

## Once You Fill Those In, Here's What I'll Design For You

### 1. 📥 Data Sources
I'll recommend specific sources based on your domain — for example:
- **Healthcare** → PubMed, clinical PDFs, patient intake audio
- **Finance** → SEC filings, earnings call transcripts, news feeds
- **Education** → Course syllabi, lecture recordings, textbook PDFs
- **Legal** → Court opinions, regulatory documents, deposition audio

### 2. 🔧 Extraction Tool Selection
I'll match tools to your actual content types:
```
PDFs with complex layouts  → Marker or Docling (not Tesseract)
Scanned/image PDFs         → Tesseract + preprocessing
Web articles               → Trafilatura (cleaner) or Crawl4AI (JS-heavy sites)
Audio/video recordings     → faster-whisper (specify model size based on volume)
```

### 3. 🧹 Cleaning Pipeline Priority
Different domains have **critically different cleaning needs**:
- **Medical** → PII removal is legally mandatory (HIPAA)
- **Legal** → Citation preservation, must NOT strip formatting
- **General research** → MinHash dedup is highest priority
- **Audio-sourced** → Disfluency removal ("um", "uh") before indexing

### 4. 🎙️ Voice Capability Decision
This depends entirely on your user workflow — which you haven't described yet. Key questions I'd answer:
- Is your user interacting hands-free? → Pipecat pipeline
- Are outputs being consumed while doing other tasks? → edge-tts or Kokoro
- Is voice just a nice-to-have? → Skip it, focus on core pipeline first

### 5. 📊 Volume & Processing

## Mini Pipeline Results

- Documents collected: 5
- Documents after cleaning: 5

## Reflections

### Data Strategy
[YOUR REFLECTION HERE]

- Which data sources are most relevant for your project?
- Which tools from this week will you actually use in your capstone?
- What's the most challenging data quality issue you expect to face?

### Pipeline Execution
[YOUR REFLECTION HERE]

- What data did you collect and how did you clean it?
- Were any documents removed by the pipeline? Why?
- How would you scale this to a full dataset for your project?

## Tools I Plan to Use

[STUDENT: List the specific tools from Week 3 you'll use in your capstone]

## Next Steps

[STUDENT: What will you build next week with RAG and vector databases?]
