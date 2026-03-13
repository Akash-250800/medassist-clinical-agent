# MedAssist — Clinical Decision Support Agent

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.74-purple?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama3_70B-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-97%25_Accuracy-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=flat-square&logo=streamlit)
![CI](https://img.shields.io/github/actions/workflow/status/YOUR_USERNAME/medassist-clinical-agent/ci.yml?style=flat-square&label=CI)

[Live Demo](https://share.streamlit.io) · [How it works](#how-it-works) · [Run it yourself](#running-it-locally)

---

I built this because I kept wondering — what would it look like if an AI agent actually *reasoned* through a patient case instead of just answering a question?

Not a chatbot. Not a search engine. Something closer to a junior doctor who reads the chart, pulls up relevant guidelines, cross-references a trained model, flags the dangerous stuff, and hands you a structured report.

That's MedAssist.

> This is a portfolio and research project. Not a medical device. Never use it instead of an actual doctor.

---

## What it does

You paste in a patient description — plain text, the way you'd actually write it — and it runs through a 7-step reasoning pipeline powered by LangGraph, Groq's Llama3 70B, a FAISS vector store, and a trained XGBoost classifier.

Here's what that looks like in practice:

**You type:**
```
72-year-old male, diabetic, smoker. Sudden crushing chest pain
radiating to jaw and left arm. Cold sweat, nausea, BP 90/60.
Medications: Metformin 500mg, Atorvastatin 20mg.
```

**It gives you:**
```
🚨 EMERGENCY — Risk Score: 0.94

ML Prediction:
  Heart Attack      89% ████████████████████
  Hypertension       7% █████
  GERD               4% ███

Why the model thinks this (SHAP):
  chest_pain     → +0.42
  sweating       → +0.28
  nausea         → +0.19
  breathlessness → +0.14

Differential Diagnosis:
  HIGH   → STEMI / Acute Coronary Syndrome
  MEDIUM → Unstable Angina
  LOW    → Aortic Dissection

Drug Warnings:
  ⚠️  Metformin + contrast dye — hold before imaging
  ⚠️  BP 90/60 — haemodynamic instability

What to do right now:
  → 12-lead ECG immediately
  → Troponin I/T + CBC + BMP
  → Cardiology consult — do not wait
```

---

## How it works

The pipeline has 7 nodes. Each one does one thing well.

```
Your input
    │
    ▼
Parse the text
Pulls out symptoms, age, medications, history
    │
    ▼
Retrieve relevant documents
Searches a FAISS index built from the Kaggle
disease-symptom dataset — 41 diseases, 131 symptoms
Returns the 5 most relevant chunks
    │
    ▼
ML Classifier
XGBoost model trained on that same dataset
Predicts top 3 diseases with confidence scores
SHAP explains exactly which symptoms drove it
    │
    ▼
Symptom analysis
The LLM reads the retrieved docs + ML output
Identifies patterns, clusters, and red flags
    │
    ▼
Differential diagnosis
Top 3–5 diagnoses with likelihood ratings
Recommended tests for each
    │
    ▼
Risk scoring
Weighted XAI score from 0.0 to 1.0
Checks for drug interactions
Labels urgency: ROUTINE / URGENT / EMERGENCY
    │
    ▼
Risk > 0.85?
    ├── YES → Emergency alert
    └── NO  → Final structured report
```

The conditional routing at the end is the part that makes LangGraph worth using here. A simple chain can't do that cleanly.

---

## The ML piece

Honestly the part I'm most proud of.

I trained an XGBoost classifier on the [Kaggle Disease-Symptom dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) — 41 diseases, 131 symptom features, binary encoding (symptom present or not). 80/20 stratified split, 5-fold cross-validation. Ended up at **97%+ test accuracy**.

But the accuracy number isn't the interesting part. The interesting part is SHAP.

Most ML models in healthcare fail not because they're wrong but because no one trusts them. A black box that says "heart attack: 87%" is useless to a clinician. SHAP makes every prediction auditable:

```
chest_pain       contributed +0.42 toward this prediction
sweating         contributed +0.28
nausea           contributed +0.19
breathlessness   contributed +0.14
```

That's what explainable AI actually means. Not a confidence score — a reason.

---

## Stack

| What | Why I chose it |
|---|---|
| **LangGraph** | State machine with conditional routing. Much cleaner than chains for multi-step reasoning. |
| **Groq + Llama3 70B** | Free, fast (under 2 seconds), genuinely good at clinical reasoning tasks. |
| **FAISS** | Local vector store, no cloud dependency, fast similarity search. |
| **HuggingFace MiniLM** | Free embeddings, GPU-accelerated, 384 dimensions — works well for symptom matching. |
| **XGBoost + SHAP** | High accuracy + full explainability. The right combination for healthcare. |
| **Streamlit** | Fastest path from working code to a shareable UI. |
| **GitHub Actions** | CI runs lint, import tests, unit tests, and a secret scanner on every push. |

---

## Running it locally

You'll need Python 3.11 and a free [Groq API key](https://console.groq.com/keys). A GPU helps for embeddings but isn't required.

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/medassist-clinical-agent.git
cd medassist-clinical-agent

# Set up environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# GPU support (skip if no NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Add your Groq key
echo GROQ_API_KEY=gsk_xxxxxxxxxxxx > .env

# Get the dataset
# Download from: kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
# Drop all 4 CSV files into: data/sample_docs/

# Build the vector index
python data/ingest.py

# Train the classifier (takes ~2 minutes)
python train_classifier.py

# Start the app
streamlit run app.py
```

Open `http://localhost:8501` and try one of the sample cases.

---

## Things worth testing

**This one triggers the emergency alert:**
```
72-year-old male, diabetic, smoker. Sudden crushing chest pain
radiating to jaw and left arm. Cold sweat, nausea. BP 90/60.
Medications: Metformin 500mg, Atorvastatin 20mg.
```

**This one tests drug interaction detection:**
```
70-year-old male on Warfarin 5mg for atrial fibrillation.
Started Ibuprofen 400mg for knee pain 1 week ago.
Now presenting with blood in urine and easy bruising.
```

**This one is a tricky neurological case:**
```
29-year-old female with sudden worst headache of her life,
neck stiffness, photophobia, fever 39C. Vomiting twice.
Non-blanching purple spots appearing on legs.
```

---

## Project layout

```
medassist-clinical-agent/
│
├── app.py                      ← Streamlit UI, 5 tabs
├── train_classifier.py         ← XGBoost training + SHAP plots
├── startup.py                  ← Rebuilds FAISS index on Streamlit Cloud
├── requirements.txt
├── packages.txt                ← Linux system deps for cloud deploy
│
├── .github/workflows/
│   └── ci.yml                  ← CI/CD pipeline
│
├── .streamlit/
│   └── config.toml             ← Theme and server config
│
├── agents/
│   └── clinical_agent.py       ← The full LangGraph pipeline lives here
│
├── rag/
│   └── retriever.py            ← FAISS + HuggingFace embeddings
│
├── utils/
│   ├── risk_scorer.py          ← XAI risk scoring logic
│   └── disease_classifier.py  ← Loads the trained model, runs SHAP
│
└── data/
    ├── ingest.py               ← Builds the FAISS index
    └── sample_docs/            ← Put the Kaggle CSVs here
```

---

## What I actually learned

**LangGraph is worth learning.** I started with a simple LangChain chain and kept running into walls when I needed branching logic. Switching to LangGraph's state machine model made the whole pipeline easier to reason about, debug, and extend.

**SHAP changes how you think about ML.** Building the explainability layer wasn't just a nice-to-have — it made me understand the model's behaviour much better. I caught a few cases where high confidence was driven by a single symptom that happened to correlate strongly in the training data.

**Groq surprised me.** I expected to use OpenAI. Tried Groq on a whim and never looked back. Llama3 70B at that speed, for free, is genuinely hard to beat for this kind of structured reasoning task.

**Deployment always finds your assumptions.** `python-magic-bin` is Windows-only. File paths behave differently. No GPU on the cloud server. Every assumption I made while developing on Windows showed up as a bug on Streamlit Cloud.

---

## Honest limitations

- 41 diseases in the training data. Real clinical practice has thousands.
- Groq free tier has rate limits — fine for a demo, not for production.
- The classifier only sees symptom presence/absence — no lab values, no imaging, no physical exam findings.
- Cold start on Streamlit Cloud takes about 60 seconds while the index rebuilds.
- This has not been validated on real patient data in any way.

---

## Where I'd take it next

- Pull live research from PubMed API instead of a static dataset
- Replace Llama3 with a fine-tuned medical LLM like BioMedLM
- Add a FHIR connector to read from real EHR systems
- Build a physician review step before the final output goes anywhere
- Add structured lab value input alongside the text
- Evaluate against clinical benchmark datasets

---

## A note on ethics

I thought carefully about the right way to frame this project.

AI in healthcare is genuinely high stakes. A wrong output from a system like this, trusted uncritically, could hurt someone. So every output includes a disclaimer. Risk scores are explained, not just shown. Nothing is stored. The whole thing is designed to support clinical thinking, not shortcut it.

If you ever build something real on top of this — please work with licensed clinicians from the start, not at the end.

---

## About me

**Akash Krishnan**
MSc Computer Science — Internationale Hochschule, Germany

I work on applied ML — forecasting, RAG systems, explainable AI, and agent pipelines. This project sits at the intersection of all of those.

[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)

---

*Built with LangGraph, Groq, FAISS, XGBoost, SHAP, HuggingFace, and Streamlit.*
*Not a medical device. For educational and portfolio use only.*
