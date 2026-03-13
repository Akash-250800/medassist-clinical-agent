import os
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class MedicalRetriever:
    """
    FAISS-based retriever for medical documents.

    Automatically loads:
      - Built-in clinical guidelines
      - Kaggle CSV dataset files from data/sample_docs/
      - Any PDF or TXT files dropped in data/sample_docs/

    Usage:
        retriever = MedicalRetriever()
        docs = retriever.retrieve("chest pain shortness of breath", k=5)
    """

    VECTOR_STORE_PATH = "data/faiss_index"
    DOCS_PATH         = "data/sample_docs"

    def __init__(self):
        self.embeddings   = self._load_embeddings()
        self.vectorstore  : Optional[FAISS] = None
        self._initialize()

    # ─────────────────────────────────────────
    # Embeddings — FREE on GTX 1650 GPU
    # ─────────────────────────────────────────

    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Load HuggingFace embedding model on GPU."""
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[RAG] Loading embeddings on: {device.upper()}")

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

    # ─────────────────────────────────────────
    # Initialize — Load or Build Index
    # ─────────────────────────────────────────

    def _initialize(self):
        """Load existing FAISS index or build from scratch."""
        index_path = Path(self.VECTOR_STORE_PATH)

        if index_path.exists():
            print("[RAG] Loading existing FAISS index...")
            self.vectorstore = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("[RAG] FAISS index loaded!")
        else:
            print("[RAG] Building new FAISS index...")
            documents = self._load_all_documents()
            self._build_index(documents)

    # ─────────────────────────────────────────
    # Document Loaders
    # ─────────────────────────────────────────

    def _load_all_documents(self) -> List[Document]:
        """Load all available documents."""
        documents = []

        # 1. Built-in clinical guidelines
        documents += self._get_builtin_guidelines()

        # 2. Kaggle CSV files
        documents += self._load_kaggle_csvs()

        # 3. PDF files from sample_docs/
        documents += self._load_pdf_files()

        # 4. TXT files from sample_docs/
        documents += self._load_txt_files()

        print(f"[RAG] Total documents loaded: {len(documents)}")
        return documents

    def _load_kaggle_csvs(self) -> List[Document]:
        """
        Load Kaggle Disease Symptom Dataset CSV files.

        Expected files in data/sample_docs/:
          - symptom_Description.csv  → disease descriptions
          - symptom_precaution.csv   → precautions per disease
          - Symptom-severity.csv     → symptom severity scores
          - dataset.csv              → disease-symptom mapping
        """
        docs_path = Path(self.DOCS_PATH)
        documents = []

        # ── symptom_Description.csv ──
        desc_file = docs_path / "symptom_Description.csv"
        if desc_file.exists():
            try:
                df = pd.read_csv(desc_file)
                df.columns = [c.strip() for c in df.columns]
                for _, row in df.iterrows():
                    disease     = str(row.get("Disease", "Unknown")).strip()
                    description = str(row.get("Description", "")).strip()
                    if description:
                        content = f"Disease: {disease}\nDescription: {description}"
                        documents.append(Document(
                            page_content=content,
                            metadata={
                                "source":   "Kaggle-symptom_Description",
                                "category": "disease_description",
                                "disease":  disease
                            }
                        ))
                print(f"[RAG] Loaded symptom_Description.csv → {len(df)} diseases")
            except Exception as e:
                print(f"[RAG] symptom_Description.csv error: {e}")

        # ── symptom_precaution.csv ──
        prec_file = docs_path / "symptom_precaution.csv"
        if prec_file.exists():
            try:
                df = pd.read_csv(prec_file)
                df.columns = [c.strip() for c in df.columns]
                for _, row in df.iterrows():
                    disease = str(row.get("Disease", "Unknown")).strip()
                    precautions = [
                        str(row.get(f"Precaution_{i}", "")).strip()
                        for i in range(1, 5)
                        if str(row.get(f"Precaution_{i}", "")).strip()
                    ]
                    if precautions:
                        content = (
                            f"Disease: {disease}\n"
                            f"Precautions:\n" +
                            "\n".join([f"- {p}" for p in precautions])
                        )
                        documents.append(Document(
                            page_content=content,
                            metadata={
                                "source":   "Kaggle-symptom_precaution",
                                "category": "precautions",
                                "disease":  disease
                            }
                        ))
                print(f"[RAG] Loaded symptom_precaution.csv → {len(df)} diseases")
            except Exception as e:
                print(f"[RAG]  symptom_precaution.csv error: {e}")

        # ── Symptom-severity.csv ──
        sev_file = docs_path / "Symptom-severity.csv"
        if sev_file.exists():
            try:
                df = pd.read_csv(sev_file)
                df.columns = [c.strip() for c in df.columns]
                # Group into batches of 10 symptoms per document
                batch_size = 10
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    lines = [
                        f"- {str(row.get('Symptom','?')).strip()}: "
                        f"severity {str(row.get('weight','?')).strip()}/7"
                        for _, row in batch.iterrows()
                    ]
                    content = "Symptom Severity Scores:\n" + "\n".join(lines)
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source":   "Kaggle-Symptom-severity",
                            "category": "severity_scores"
                        }
                    ))
                print(f"[RAG] Loaded Symptom-severity.csv → {len(df)} symptoms")
            except Exception as e:
                print(f"[RAG] Symptom-severity.csv error: {e}")

        # ── dataset.csv ──
        data_file = docs_path / "dataset.csv"
        if data_file.exists():
            try:
                df = pd.read_csv(data_file)
                df.columns = [c.strip() for c in df.columns]

                # Group by disease and collect symptoms
                if "Disease" in df.columns:
                    for disease, group in df.groupby("Disease"):
                        symptom_cols = [
                            c for c in df.columns
                            if c.lower().startswith("symptom")
                        ]
                        symptoms = []
                        for col in symptom_cols:
                            vals = group[col].dropna().unique()
                            symptoms.extend([
                                v.strip() for v in vals
                                if str(v).strip() and str(v).strip() != "nan"
                            ])
                        symptoms = list(set(symptoms))[:20]

                        if symptoms:
                            content = (
                                f"Disease: {disease}\n"
                                f"Associated Symptoms:\n" +
                                "\n".join([f"- {s}" for s in symptoms])
                            )
                            documents.append(Document(
                                page_content=content,
                                metadata={
                                    "source":   "Kaggle-dataset",
                                    "category": "disease_symptoms",
                                    "disease":  str(disease)
                                }
                            ))
                print(f"[RAG] Loaded dataset.csv → {df['Disease'].nunique()} unique diseases")
            except Exception as e:
                print(f"[RAG] dataset.csv error: {e}")

        if not documents:
            print("[RAG]  No Kaggle CSV files found in data/sample_docs/")
            print("[RAG]     Download from: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")

        return documents

    def _load_pdf_files(self) -> List[Document]:
        """Load PDF files from sample_docs/."""
        docs_path = Path(self.DOCS_PATH)
        documents = []
        for pdf_path in docs_path.glob("**/*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages  = loader.load()
                for page in pages:
                    page.metadata["source"]   = pdf_path.name
                    page.metadata["category"] = "pdf_document"
                documents.extend(pages)
                print(f"[RAG] Loaded PDF: {pdf_path.name} ({len(pages)} pages)")
            except Exception as e:
                print(f"[RAG] PDF error {pdf_path.name}: {e}")
        return documents

    def _load_txt_files(self) -> List[Document]:
        """Load TXT files from sample_docs/."""
        docs_path = Path(self.DOCS_PATH)
        documents = []
        for txt_path in docs_path.glob("**/*.txt"):
            try:
                loader = TextLoader(str(txt_path), encoding="utf-8")
                docs   = loader.load()
                for doc in docs:
                    doc.metadata["source"]   = txt_path.name
                    doc.metadata["category"] = "txt_document"
                documents.extend(docs)
                print(f"[RAG] Loaded TXT: {txt_path.name}")
            except Exception as e:
                print(f"[RAG] TXT error {txt_path.name}: {e}")
        return documents

    def _get_builtin_guidelines(self) -> List[Document]:
        """Built-in clinical guidelines — always included."""
        samples = [
            {
                "content": """Chest Pain Clinical Guidelines (WHO 2023):
Red flags requiring emergency evaluation:
- Crushing/pressure chest pain radiating to arm, jaw, or back
- Associated diaphoresis, nausea, shortness of breath
- Sudden onset severe chest pain (aortic dissection)
Key differentials: AMI, Unstable Angina, PE, Aortic Dissection, Pneumothorax
Initial workup: ECG within 10 minutes, Troponin I/T, CXR, CBC, BMP
Risk scores: TIMI for ACS, Wells for PE""",
                "source": "WHO Clinical Guidelines - Chest Pain 2023",
                "category": "cardiology"
            },
            {
                "content": """Diabetes Management (ADA Standards 2024):
Diagnostic Criteria: Fasting glucose >= 126, HbA1c >= 6.5%, OGTT >= 200
Complications risk: Poor glycemic control, Hypertension, Dyslipidemia, Obesity
Management: Metformin first-line, GLP-1 for CV benefit, SGLT2 for renal protection
Monitoring: HbA1c every 3 months, annual eye/foot/kidney screening
Critical warning: Metformin contraindicated if eGFR < 30""",
                "source": "ADA Standards of Care 2024",
                "category": "endocrinology"
            },
            {
                "content": """Hypertension Management (ESC Guidelines 2023):
Classification: Grade 1: 140-159/90-99, Grade 2: 160-179/100-109, Grade 3: >=180/110
Hypertensive Crisis: >180/120 with organ damage
Treatment target: BP < 130/80 mmHg for all grades
First-line drugs: ACE inhibitors/ARBs (especially with diabetes/CKD)
Drug interaction: NSAIDs reduce antihypertensive efficacy""",
                "source": "ESC Hypertension Guidelines 2023",
                "category": "cardiology"
            },
            {
                "content": """Stroke Recognition (AHA/ASA 2023) — FAST Protocol:
F - Face drooping, A - Arm weakness, S - Speech difficulty, T - Time critical
IV tPA window: within 4.5 hours of symptom onset
Thrombectomy: within 24 hours for large vessel occlusion
Thunderclap headache = subarachnoid hemorrhage until proven otherwise
SNOOP red flags: Systemic symptoms, Neurological deficits, Onset sudden, Older age""",
                "source": "AHA/ASA Stroke Guidelines 2023",
                "category": "neurology"
            },
            {
                "content": """Sepsis Recognition (Surviving Sepsis Campaign 2023):
qSOFA score: RR>=22 + altered mentation + SBP<=100 = high risk
Septic shock: hypotension + lactate >2 despite fluids + vasopressors needed
Hour-1 Bundle: lactate, blood cultures, antibiotics, 30mL/kg fluid, vasopressors
Critical: broad-spectrum antibiotics within 1 hour of recognition""",
                "source": "Surviving Sepsis Campaign 2023",
                "category": "critical_care"
            },
            {
                "content": """Drug Interactions — Critical Combinations (WHO 2023):
1. Warfarin + NSAIDs: Major bleeding risk — AVOID
2. Metformin + Contrast dye: Hold 48h before/after — AKI risk
3. SSRIs + MAOIs: Serotonin syndrome — FATAL — 14-day washout
4. Statins + CYP3A4 inhibitors: Rhabdomyolysis risk
5. Digoxin + Amiodarone: Toxicity — reduce digoxin 50%
6. ACE inhibitors + K-sparing diuretics: Hyperkalemia risk""",
                "source": "WHO Drug Interaction Guidelines 2023",
                "category": "pharmacology"
            },
            {
                "content": """COPD and Respiratory Assessment (GOLD 2024):
Acute dyspnea red flags: SpO2 <90%, RR >30/min, accessory muscles, altered consciousness
COPD diagnosis: FEV1/FVC <0.70 post-bronchodilator
Pneumonia signs: fever, productive cough, consolidation on CXR
Heart failure signs: bilateral crackles, peripheral edema, orthopnea, elevated BNP
PE signs: sudden pleuritic pain, tachycardia, risk factors (immobility, cancer, DVT)""",
                "source": "GOLD COPD Guidelines 2024",
                "category": "pulmonology"
            },
            {
                "content": """Acute Kidney Injury — KDIGO 2023:
AKI defined: creatinine rise >=0.3 mg/dL in 48h OR >=1.5x baseline in 7 days
Staging: Stage 1: 1.5-1.9x, Stage 2: 2.0-2.9x, Stage 3: >=3x baseline
Most common cause: Pre-renal (60-70%) — dehydration, hypoperfusion
Nephrotoxins to avoid: NSAIDs, aminoglycosides, contrast dye
Metformin: contraindicated if eGFR <30 ml/min""",
                "source": "KDIGO AKI Guidelines 2023",
                "category": "nephrology"
            },
        ]

        documents = []
        for sample in samples:
            documents.append(Document(
                page_content=sample["content"].strip(),
                metadata={
                    "source":   sample["source"],
                    "category": sample["category"]
                }
            ))
        print(f"[RAG] Loaded {len(documents)} built-in clinical guidelines")
        return documents

    # ─────────────────────────────────────────
    # Build FAISS Index
    # ─────────────────────────────────────────

    def _build_index(self, documents: List[Document]):
        """Chunk documents and build FAISS vector index."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[RAG] Created {len(chunks)} chunks")

        print("[RAG] Building FAISS index on GPU...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        os.makedirs(self.VECTOR_STORE_PATH, exist_ok=True)
        self.vectorstore.save_local(self.VECTOR_STORE_PATH)
        print(f"[RAG] FAISS index saved to {self.VECTOR_STORE_PATH}")

    # ─────────────────────────────────────────
    # Retrieve
    # ─────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k relevant clinical documents for a query.

        Args:
            query : Patient description or clinical query
            k     : Number of documents to retrieve

        Returns:
            List of dicts with content, source, category, relevance_score
        """
        if not self.vectorstore:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        retrieved = []
        for doc, score in results:
            retrieved.append({
                "content":        doc.page_content,
                "source":         doc.metadata.get("source", "Unknown"),
                "category":       doc.metadata.get("category", "general"),
                "disease":        doc.metadata.get("disease", ""),
                "relevance_score": round(float(1 / (1 + score)), 3)
            })

        return retrieved

    def add_document(self, content: str, source: str, category: str = "general"):
        """Add a new document to the vector store at runtime."""
        doc = Document(
            page_content=content,
            metadata={"source": source, "category": category}
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents([doc])
        self.vectorstore.add_documents(chunks)
        self.vectorstore.save_local(self.VECTOR_STORE_PATH)
        print(f"[RAG] Added document: {source}")
