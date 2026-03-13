"""
MedAssist — Data Ingestion Pipeline
=====================================
Loads all documents, builds FAISS index.
Run this ONCE before starting the app.

Usage:
    python data\ingest.py
    python data\ingest.py --rebuild
    python data\ingest.py --test
"""
import os
import sys
import time
import shutil
import argparse
import pandas as pd
from pathlib import Path
from typing import List

# ── Fix import paths ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # Always run from project root

print("=" * 60)
print("   MedAssist — Data Ingestion Pipeline")
print("=" * 60)
print(f"   Project Root : {ROOT}")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Paths ──
DOCS_DIR  = ROOT / "data" / "sample_docs"
INDEX_DIR = ROOT / "data" / "faiss_index"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

print(f"   Docs folder  : {DOCS_DIR}")
print(f"   Index folder : {INDEX_DIR}")
print("=" * 60)


# ─────────────────────────────────────────────
# Step 1 — Load Embedding Model
# ─────────────────────────────────────────────

def load_embeddings():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[EMBED] Device : {device.upper()}")
    print(f"[EMBED] Model  : all-MiniLM-L6-v2")
    print(f"[EMBED] Loading... (first run downloads ~90MB)")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"[EMBED] Model loaded on {device.upper()}!")
    return embeddings


# ─────────────────────────────────────────────
# Step 2 — Load Documents
# ─────────────────────────────────────────────

def load_builtin_guidelines() -> List[Document]:
    """Built-in clinical guidelines — always loaded."""
    print("\n[DATA] Loading built-in clinical guidelines...")
    samples = [
        {
            "content": """Chest Pain Clinical Guidelines (WHO 2023):
Red flags requiring emergency evaluation:
- Crushing chest pain radiating to arm, jaw, or back
- Diaphoresis, nausea, shortness of breath
- Sudden onset severe chest pain
Key differentials: AMI, Unstable Angina, PE, Aortic Dissection
Workup: ECG within 10 min, Troponin, CXR, CBC, BMP
Risk scores: TIMI for ACS, Wells for PE""",
            "source": "WHO Clinical Guidelines - Chest Pain 2023",
            "category": "cardiology"
        },
        {
            "content": """Diabetes Management (ADA Standards 2024):
Diagnostic: Fasting glucose>=126, HbA1c>=6.5%, OGTT>=200
Management: Metformin first-line, GLP-1 for CV benefit
SGLT2 inhibitors for renal protection
Monitoring: HbA1c every 3 months, annual eye/foot/kidney
WARNING: Metformin contraindicated if eGFR < 30""",
            "source": "ADA Standards of Care 2024",
            "category": "endocrinology"
        },
        {
            "content": """Hypertension Management (ESC 2023):
Grade 1: 140-159/90-99, Grade 2: 160-179/100-109
Grade 3: >=180/110, Crisis: >180/120 with organ damage
Target: BP < 130/80 mmHg for all grades
First-line: ACE inhibitors/ARBs especially with diabetes/CKD
Warning: NSAIDs reduce antihypertensive efficacy""",
            "source": "ESC Hypertension Guidelines 2023",
            "category": "cardiology"
        },
        {
            "content": """Stroke Recognition (AHA/ASA 2023) FAST Protocol:
F-Face drooping, A-Arm weakness, S-Speech difficulty, T-Time critical
IV tPA: within 4.5 hours of symptom onset
Thrombectomy: within 24 hours for large vessel occlusion
Thunderclap headache = subarachnoid hemorrhage until proven otherwise
SNOOP red flags: Sudden onset, Neurological deficits, Older age >50""",
            "source": "AHA/ASA Stroke Guidelines 2023",
            "category": "neurology"
        },
        {
            "content": """Sepsis (Surviving Sepsis Campaign 2023):
qSOFA: RR>=22 + altered mentation + SBP<=100 = high risk
Septic shock: hypotension + lactate >2 despite fluids
Hour-1 Bundle: blood cultures, antibiotics within 1 hour
30mL/kg crystalloid for hypotension, vasopressors if needed""",
            "source": "Surviving Sepsis Campaign 2023",
            "category": "critical_care"
        },
        {
            "content": """Drug Interactions (WHO 2023):
1. Warfarin + NSAIDs: Major bleeding risk AVOID
2. Metformin + Contrast: Hold 48h before/after AKI risk
3. SSRIs + MAOIs: Serotonin syndrome FATAL 14-day washout
4. Statins + CYP3A4 inhibitors: Rhabdomyolysis risk
5. Digoxin + Amiodarone: Toxicity reduce digoxin 50%
6. ACE inhibitors + K-sparing diuretics: Hyperkalemia""",
            "source": "WHO Drug Interaction Guidelines 2023",
            "category": "pharmacology"
        },
        {
            "content": """COPD/Respiratory (GOLD 2024):
Acute red flags: SpO2 <90%, RR >30/min, accessory muscles
COPD: FEV1/FVC <0.70 post-bronchodilator
Pneumonia: fever, productive cough, consolidation on CXR
Heart failure: bilateral crackles, edema, orthopnea, elevated BNP
PE: sudden pleuritic pain, tachycardia, risk factors""",
            "source": "GOLD COPD Guidelines 2024",
            "category": "pulmonology"
        },
        {
            "content": """Acute Kidney Injury (KDIGO 2023):
AKI: creatinine rise >=0.3 mg/dL in 48h or >=1.5x baseline
Staging: Stage 1:1.5-1.9x, Stage 2:2.0-2.9x, Stage 3:>=3x
Most common cause: Pre-renal dehydration or hypoperfusion
Avoid: NSAIDs, aminoglycosides, contrast dye in AKI
Metformin contraindicated if eGFR <30""",
            "source": "KDIGO AKI Guidelines 2023",
            "category": "nephrology"
        },
    ]
    docs = []
    for i, s in enumerate(samples, 1):
        docs.append(Document(
            page_content=s["content"].strip(),
            metadata={"source": s["source"], "category": s["category"]}
        ))
        print(f"    [{i}/{len(samples)}] {s['source']}")
    return docs


def load_kaggle_csvs() -> List[Document]:
    """Load Kaggle Disease Symptom Dataset CSV files."""
    print("\n[DATA] Loading Kaggle CSV files...")
    docs = []

    # ── symptom_Description.csv ──
    f = DOCS_DIR / "symptom_Description.csv"
    if f.exists():
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            count = 0
            for _, row in df.iterrows():
                disease = str(row.get("Disease", "")).strip()
                desc    = str(row.get("Description", "")).strip()
                if disease and desc and desc != "nan":
                    docs.append(Document(
                        page_content=f"Disease: {disease}\nDescription: {desc}",
                        metadata={"source": "Kaggle-symptom_Description", "category": "disease_description", "disease": disease}
                    ))
                    count += 1
            print(f"    symptom_Description.csv → {count} diseases loaded")
        except Exception as e:
            print(f"    symptom_Description.csv error: {e}")
    else:
        print(f"   symptom_Description.csv not found in {DOCS_DIR}")

    # ── symptom_precaution.csv ──
    f = DOCS_DIR / "symptom_precaution.csv"
    if f.exists():
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            count = 0
            for _, row in df.iterrows():
                disease = str(row.get("Disease", "")).strip()
                precs   = [str(row.get(f"Precaution_{i}", "")).strip()
                           for i in range(1, 5)
                           if str(row.get(f"Precaution_{i}", "")).strip() not in ["", "nan"]]
                if disease and precs:
                    content = f"Disease: {disease}\nPrecautions:\n" + "\n".join([f"- {p}" for p in precs])
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": "Kaggle-symptom_precaution", "category": "precautions", "disease": disease}
                    ))
                    count += 1
            print(f"    symptom_precaution.csv → {count} diseases loaded")
        except Exception as e:
            print(f"    symptom_precaution.csv error: {e}")
    else:
        print(f"    symptom_precaution.csv not found in {DOCS_DIR}")

    # ── Symptom-severity.csv ──
    f = DOCS_DIR / "Symptom-severity.csv"
    if f.exists():
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            lines = []
            for _, row in df.iterrows():
                sym = str(row.get("Symptom", "")).strip()
                sev = str(row.get("weight", "")).strip()
                if sym and sev and sym != "nan":
                    lines.append(f"- {sym}: severity {sev}/7")
            # Split into batches of 15
            for i in range(0, len(lines), 15):
                batch = lines[i:i+15]
                docs.append(Document(
                    page_content="Symptom Severity Scores:\n" + "\n".join(batch),
                    metadata={"source": "Kaggle-Symptom-severity", "category": "severity_scores"}
                ))
            print(f"   Symptom-severity.csv → {len(lines)} symptoms loaded")
        except Exception as e:
            print(f"   Symptom-severity.csv error: {e}")
    else:
        print(f"    Symptom-severity.csv not found in {DOCS_DIR}")

    # ── dataset.csv ──
    f = DOCS_DIR / "dataset.csv"
    if f.exists():
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            count = 0
            if "Disease" in df.columns:
                symptom_cols = [c for c in df.columns if "Symptom" in c]
                for disease, group in df.groupby("Disease"):
                    symptoms = []
                    for col in symptom_cols:
                        vals = group[col].dropna().unique()
                        symptoms.extend([
                            v.strip() for v in vals
                            if str(v).strip() not in ["", "nan"]
                        ])
                    symptoms = list(set(symptoms))[:20]
                    if symptoms:
                        content = f"Disease: {disease}\nAssociated Symptoms:\n" + "\n".join([f"- {s}" for s in symptoms])
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": "Kaggle-dataset", "category": "disease_symptoms", "disease": str(disease)}
                        ))
                        count += 1
            print(f"   dataset.csv → {count} diseases loaded")
        except Exception as e:
            print(f"    dataset.csv error: {e}")
    else:
        print(f"    dataset.csv not found in {DOCS_DIR}")

    return docs


def load_pdf_files() -> List[Document]:
    docs = []
    for f in DOCS_DIR.glob("**/*.pdf"):
        try:
            pages = PyPDFLoader(str(f)).load()
            for p in pages:
                p.metadata.update({"source": f.name, "category": "pdf"})
            docs.extend(pages)
            print(f"   PDF: {f.name} ({len(pages)} pages)")
        except Exception as e:
            print(f"   PDF {f.name}: {e}")
    return docs


def load_txt_files() -> List[Document]:
    docs = []
    for f in DOCS_DIR.glob("**/*.txt"):
        try:
            d = TextLoader(str(f), encoding="utf-8").load()
            for doc in d:
                doc.metadata.update({"source": f.name, "category": "txt"})
            docs.extend(d)
            print(f"   TXT: {f.name}")
        except Exception as e:
            print(f"   XT {f.name}: {e}")
    return docs


# ─────────────────────────────────────────────
# Step 3 — Chunk Documents
# ─────────────────────────────────────────────

def chunk_documents(documents: List[Document]) -> List[Document]:
    print(f"\n[CHUNK] Splitting {len(documents)} documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"[CHUNK] Created {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# Step 4 — Build FAISS Index
# ─────────────────────────────────────────────

def build_faiss_index(chunks, embeddings, force_rebuild=False):
    if force_rebuild and INDEX_DIR.exists():
        print(f"\n[FAISS] Force rebuild — deleting old index...")
        shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()) and not force_rebuild:
        print(f"\n[FAISS] Loading existing index from {INDEX_DIR}...")
        vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        print(f"[FAISS] Existing index loaded!")
        return vs

    print(f"\n[FAISS] Building FAISS index ({len(chunks)} chunks)...")
    start = time.time()
    vs = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - start
    vs.save_local(str(INDEX_DIR))
    print(f"[FAISS] Index built in {elapsed:.1f}s and saved to {INDEX_DIR}")
    return vs


# ─────────────────────────────────────────────
# Step 5 — Test Retrieval
# ─────────────────────────────────────────────

def test_retrieval(vectorstore):
    print("\n" + "=" * 60)
    print("   Testing RAG Retrieval...")
    print("=" * 60)
    queries = [
        "chest pain radiating to arm",
        "diabetes HbA1c management",
        "drug interaction warfarin",
    ]
    for q in queries:
        results = vectorstore.similarity_search(q, k=2)
        print(f"\nQuery: '{q}'")
        for i, doc in enumerate(results, 1):
            src     = doc.metadata.get("source", "Unknown")
            preview = doc.page_content[:100].replace("\n", " ").strip()
            print(f"   [{i}] {src}")
            print(f"       {preview}...")
    print("\n[TEST] Retrieval working correctly!")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedAssist Ingestion Pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild FAISS index")
    parser.add_argument("--test",    action="store_true", help="Test retrieval after ingestion")
    args = parser.parse_args()

    # Load all documents
    all_docs = []
    builtin = load_builtin_guidelines()
    kaggle  = load_kaggle_csvs()
    pdfs    = load_pdf_files()
    txts    = load_txt_files()

    all_docs = builtin + kaggle + pdfs + txts

    print(f"\n[DATA] ── Summary ──────────────────────")
    print(f"[DATA]   Built-in guidelines : {len(builtin)}")
    print(f"[DATA]   Kaggle CSV docs     : {len(kaggle)}")
    print(f"[DATA]   PDF files           : {len(pdfs)}")
    print(f"[DATA]   TXT files           : {len(txts)}")
    print(f"[DATA]   TOTAL               : {len(all_docs)}")

    # Chunk
    chunks = chunk_documents(all_docs)

    # Load embeddings
    embeddings = load_embeddings()

    # Build FAISS
    vectorstore = build_faiss_index(chunks, embeddings, force_rebuild=args.rebuild)

    # Summary
    print("\n" + "=" * 60)
    print("   Ingestion Complete!")
    print("=" * 60)
    print(f"   Documents  : {len(all_docs)}")
    print(f"   Chunks     : {len(chunks)}")
    print(f"   Index path : {INDEX_DIR}")
    print("=" * 60)
    print("\n   Now run: streamlit run app.py")
    print("=" * 60)

    # Optional test
    if args.test:
        test_retrieval(vectorstore)


if __name__ == "__main__":
    main()
