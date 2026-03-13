"""
MedAssist - Clinical Decision Support Agent
=============================================
LangGraph RAG + Reasoning Agent for Healthcare
LLM  : Groq API (FREE) — Llama3 70B
RAG  : FAISS + HuggingFace Embeddings (GPU)
Data : Kaggle Disease Symptom Dataset

LangGraph Flow:
  parse_input → retrieve_context → analyze_symptoms
  → support_diagnosis → flag_risks
  → [conditional] → generate_report / emergency_alert → END

Python : 3.11.4
GPU    : GTX 1650 (CUDA 12.8)
"""

import os
import sys
from pathlib import Path

# Fix import paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

from rag.retriever import MedicalRetriever
from utils.risk_scorer import RiskScorer
from utils.disease_classifier import DiseaseClassifier


# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────

class ClinicalState(TypedDict):
    """
    Shared state dictionary passed between all LangGraph nodes.
    Each node reads from and writes to this state.
    """
    messages        : Annotated[list, add_messages]
    patient_input   : str
    retrieved_docs  : List[dict]
    symptom_analysis: Optional[str]
    diagnosis_support: Optional[str]
    risk_flags      : Optional[List[str]]
    risk_score      : Optional[float]
    final_response  : Optional[str]
    confidence      : Optional[float]
    classifier_result: Optional[dict]


# ─────────────────────────────────────────────
# LLM Setup — Groq (FREE)
# ─────────────────────────────────────────────

def get_llm(temperature: float = 0.2) -> ChatGroq:
    """
    Initialize Groq LLM.
    Free limits: 30 req/min, 14,400 req/day

    Models:
      llama3-70b-8192     → Best quality (recommended)
      llama3-8b-8192      → Faster, lighter
      mixtral-8x7b-32768  → Longest context (32K)
      gemma2-9b-it        → Google Gemma2
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found!\n"
            "Add to .env file: GROQ_API_KEY=gsk_xxxx\n"
            "Get free key at: https://console.groq.com/keys"
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=api_key,
        max_tokens=2048,
    )


# ─────────────────────────────────────────────
# Node 1: Input Parser
# ─────────────────────────────────────────────

def parse_patient_input(state: ClinicalState) -> ClinicalState:
    """Node 1 — Parse and structure raw patient input."""
    llm = get_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a medical input parser AI.
Extract structured information from patient reports.
Identify: symptoms, age, gender, duration, medical_history, medications.
Summarize clearly. If information is missing, note as 'Not mentioned'.
Keep response concise and clinical."""),
        HumanMessage(content=f"Patient Report:\n{state['patient_input']}")
    ])

    response = llm.invoke(prompt.format_messages())
    state["messages"] = state.get("messages", []) + [
        HumanMessage(content=state["patient_input"]),
        AIMessage(content=f"[Input Parsed]\n{response.content}")
    ]
    return state


# ─────────────────────────────────────────────
# Node 2: RAG Retrieval
# ─────────────────────────────────────────────

def retrieve_medical_context(state: ClinicalState) -> ClinicalState:
    """Node 2 — FAISS RAG retrieval from Kaggle + built-in data."""
    retriever = MedicalRetriever()
    docs = retriever.retrieve(state["patient_input"], k=5)
    state["retrieved_docs"] = docs
    state["messages"] = state.get("messages", []) + [
        AIMessage(content=f"[RAG] Retrieved {len(docs)} relevant clinical documents.")
    ]
    return state



# ─────────────────────────────────────────────
# Node 2.5: ML Disease Classifier (XGBoost + SHAP)
# ─────────────────────────────────────────────

def classify_disease(state: ClinicalState) -> ClinicalState:
    """
    Node 2.5 — Trained XGBoost Disease Classifier.

    Uses the model trained on Kaggle dataset to:
      - Extract symptoms from patient text
      - Predict top-3 most likely diseases
      - Generate SHAP explanation for transparency
    Runs BEFORE LLM symptom analysis to provide
    data-driven predictions alongside LLM reasoning.
    """
    clf    = DiseaseClassifier()
    result = clf.predict(state["patient_input"], top_k=3)

    state["classifier_result"] = result

    if result.get("classifier_available"):
        preds = result.get("top_predictions", [])
        pred_text = " | ".join([
            f"{p['disease']} ({p['confidence']:.0%})"
            for p in preds
        ])
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=(
                f"[ML Classifier] Top predictions: {pred_text}\n"
                f"Symptoms detected: {len(result.get('detected_symptoms', []))} | "
                f"Model accuracy: {result.get('model_accuracy', 0):.1%}"
            ))
        ]
    else:
        state["messages"] = state.get("messages", []) + [
            AIMessage(content="[ML Classifier] Not available — run train_classifier.py")
        ]

    return state

# ─────────────────────────────────────────────
# Node 3: Symptom Analyzer
# ─────────────────────────────────────────────

def analyze_symptoms(state: ClinicalState) -> ClinicalState:
    """Node 3 — Evidence-based symptom analysis using retrieved docs."""
    llm = get_llm(temperature=0.1)

    context = "\n\n".join([
        f"[Source: {doc.get('source', 'Unknown')} | "
        f"Category: {doc.get('category', 'general')}]\n"
        f"{doc.get('content', '')}"
        for doc in state.get("retrieved_docs", [])
    ])

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a clinical symptom analysis AI (decision SUPPORT only).
Using provided clinical guidelines and patient data:
1. List PRIMARY symptoms identified
2. List SECONDARY symptoms
3. Identify symptom clusters or patterns
4. Flag any RED FLAG symptoms (emergency indicators)
5. Note relevant clinical context from guidelines

Be evidence-based. Cite sources when relevant.
IMPORTANT: This is decision support only — not a diagnosis."""),
        HumanMessage(content=f"""Patient Report:
{state['patient_input']}

ML Classifier Predictions (XGBoost trained on Kaggle dataset):
{state.get('classifier_result', {}).get('shap_explanation', 'Not available')}

Retrieved Clinical Guidelines:
{context if context else 'No additional context retrieved.'}

Provide symptom analysis:""")
    ])

    response = llm.invoke(prompt.format_messages())
    state["symptom_analysis"] = response.content
    state["messages"] = state.get("messages", []) + [
        AIMessage(content=f"[Symptom Analysis]\n{response.content}")
    ]
    return state


# ─────────────────────────────────────────────
# Node 4: Diagnosis Support
# ─────────────────────────────────────────────

def support_diagnosis(state: ClinicalState) -> ClinicalState:
    """Node 4 — Differential diagnosis generation."""
    llm = get_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a clinical decision support AI.
Based on symptom analysis provide differential diagnosis support.

For each diagnosis provide:
1. Condition name
2. Likelihood: High / Medium / Low
3. Supporting evidence (which symptoms match)
4. Recommended diagnostic tests
5. Immediate next steps

List top 3-5 differential diagnoses only.
Always remind: This is AI decision SUPPORT, not a final diagnosis."""),
        HumanMessage(content=f"""Patient Report:
{state['patient_input']}

Symptom Analysis:
{state.get('symptom_analysis', 'Not available')}

Provide differential diagnosis support:""")
    ])

    response = llm.invoke(prompt.format_messages())
    state["diagnosis_support"] = response.content
    state["messages"] = state.get("messages", []) + [
        AIMessage(content=f"[Diagnosis Support]\n{response.content}")
    ]
    return state


# ─────────────────────────────────────────────
# Node 5: Risk Flagging
# ─────────────────────────────────────────────

def flag_risks(state: ClinicalState) -> ClinicalState:
    """Node 5 — Clinical risk assessment with XAI scoring."""
    llm = get_llm(temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a clinical risk assessment AI.
Return your response in this EXACT format:
RED FLAGS: [list each red flag on new line with - prefix]
RISK FACTORS: [list each risk factor on new line with - prefix]
DRUG WARNINGS: [list any drug interactions or warnings]
URGENCY: [ONE of: ROUTINE / URGENT / EMERGENCY]
CONFIDENCE: [number between 0.0 and 1.0]

Be specific and evidence-based. Err on the side of caution."""),
        HumanMessage(content=f"""Patient Report:
{state['patient_input']}

Symptom Analysis:
{state.get('symptom_analysis', 'Not available')}

Diagnosis Support:
{state.get('diagnosis_support', 'Not available')}

Assess all clinical risks:""")
    ])

    response = llm.invoke(prompt.format_messages())

    scorer = RiskScorer()
    risk_data = scorer.parse_and_score(response.content)

    state["risk_flags"] = risk_data.get("red_flags", [])
    state["risk_score"] = risk_data.get("risk_score", 0.0)
    state["confidence"] = risk_data.get("confidence", 0.5)
    urgency = risk_data.get("urgency", "ROUTINE")

    state["messages"] = state.get("messages", []) + [
        AIMessage(
            content=f"[Risk Assessment] "
                    f"Score: {state['risk_score']:.2f} | "
                    f"Urgency: {urgency} | "
                    f"Confidence: {state['confidence']:.0%}"
        )
    ]
    return state


# ─────────────────────────────────────────────
# Node 6: Final Report Generator
# ─────────────────────────────────────────────

def generate_final_response(state: ClinicalState) -> ClinicalState:
    """Node 6 — Synthesize full structured clinical decision support report."""
    llm = get_llm(temperature=0.2)

    risk_score = state.get("risk_score", 0.0)
    risk_label = (
        "HIGH RISK"   if risk_score > 0.7 else
        "MEDIUM RISK" if risk_score > 0.4 else
        "LOW RISK"
    )

    sources = list(set([
        doc.get("source", "Unknown")
        for doc in state.get("retrieved_docs", [])
    ]))

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are generating a structured clinical decision support report
for review by an attending physician.

Format the report with these exact sections:
1. EXECUTIVE SUMMARY
2. SYMPTOM ANALYSIS
3. DIFFERENTIAL DIAGNOSES
4. RISK ASSESSMENT
5. RECOMMENDED ACTIONS
6. EVIDENCE SOURCES
7. DISCLAIMER

Be concise, professional and clinical.
Include confidence scores for explainability."""),
        HumanMessage(content=f"""
Symptom Analysis:
{state.get('symptom_analysis', 'Not available')}

Diagnosis Support:
{state.get('diagnosis_support', 'Not available')}

Risk Level    : {risk_label}
Risk Score    : {risk_score:.0%}
Risk Flags    : {', '.join(state.get('risk_flags', [])) or 'None identified'}
Confidence    : {state.get('confidence', 0.5):.0%}
Sources Used  : {', '.join(sources) or 'Built-in guidelines'}
Docs Retrieved: {len(state.get('retrieved_docs', []))}

Generate the complete clinical decision support report:""")
    ])

    response = llm.invoke(prompt.format_messages())
    state["final_response"] = response.content
    state["messages"] = state.get("messages", []) + [
        AIMessage(content=response.content)
    ]
    return state


# ─────────────────────────────────────────────
# Emergency Alert Node
# ─────────────────────────────────────────────

def emergency_alert(state: ClinicalState) -> ClinicalState:
    """Emergency fast-track — triggered when risk_score > 0.85."""
    flags = state.get("risk_flags", [])
    flags_text = "\n".join([f"  🚨 {f}" for f in flags]) if flags else "  🚨 Critical indicators detected"

    state["final_response"] = f"""
╔══════════════════════════════════════════╗
║         🚨 EMERGENCY ALERT 🚨            ║
╚══════════════════════════════════════════╝

RISK SCORE  : {state.get('risk_score', 0):.0%}
CONFIDENCE  : {state.get('confidence', 0):.0%}
URGENCY     : EMERGENCY

CRITICAL FLAGS DETECTED:
{flags_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED IMMEDIATE ACTION:
  → Contact emergency services (112/911)
  → Escalate to attending physician NOW
  → Do NOT wait for routine assessment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ This is an AI-generated alert.
   Clinical judgment by a licensed physician
   is ALWAYS required.
"""
    return state


# ─────────────────────────────────────────────
# Conditional Router
# ─────────────────────────────────────────────

def route_by_risk(state: ClinicalState) -> str:
    """Route to emergency if risk > 0.85, else generate normal report."""
    return "emergency_alert" if state.get("risk_score", 0.0) > 0.85 else "generate_final_response"


# ─────────────────────────────────────────────
# Build LangGraph
# ─────────────────────────────────────────────

def build_clinical_graph() -> StateGraph:
    """Compile the full LangGraph clinical decision support pipeline."""
    graph = StateGraph(ClinicalState)

    graph.add_node("parse_input",             parse_patient_input)
    graph.add_node("classify_disease",        classify_disease)
    graph.add_node("retrieve_context",        retrieve_medical_context)
    graph.add_node("analyze_symptoms",        analyze_symptoms)
    graph.add_node("support_diagnosis",       support_diagnosis)
    graph.add_node("flag_risks",              flag_risks)
    graph.add_node("generate_final_response", generate_final_response)
    graph.add_node("emergency_alert",         emergency_alert)

    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input",        "retrieve_context")
    graph.add_edge("retrieve_context",   "classify_disease")
    graph.add_edge("classify_disease",   "analyze_symptoms")
    graph.add_edge("analyze_symptoms",   "support_diagnosis")
    graph.add_edge("support_diagnosis",  "flag_risks")

    graph.add_conditional_edges(
        "flag_risks",
        route_by_risk,
        {
            "generate_final_response": "generate_final_response",
            "emergency_alert":         "emergency_alert"
        }
    )

    graph.add_edge("generate_final_response", END)
    graph.add_edge("emergency_alert",         END)

    return graph.compile()


# ─────────────────────────────────────────────
# Public Entry Point
# ─────────────────────────────────────────────

def run_clinical_agent(patient_input: str) -> dict:
    """
    Run the full MedAssist pipeline.

    Args:
        patient_input: Free-text patient report

    Returns:
        State dict with final_response, risk_score,
        risk_flags, confidence, retrieved_docs
    """
    graph = build_clinical_graph()

    initial_state: ClinicalState = {
        "messages":          [],
        "patient_input":     patient_input,
        "retrieved_docs":    [],
        "symptom_analysis":  None,
        "diagnosis_support": None,
        "risk_flags":        [],
        "risk_score":        0.0,
        "final_response":    None,
        "confidence":        0.0
    }

    return graph.invoke(initial_state)


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing MedAssist with Groq Llama3 70B...\n")
    test_input = """
    45-year-old male with crushing chest pain radiating
    to left arm for 2 hours, shortness of breath, sweating.
    History: hypertension, smoking.
    Medications: Lisinopril 10mg, Aspirin 81mg.
    """
    result = run_clinical_agent(test_input)
    print(result["final_response"])
    print(f"\nRisk Score : {result['risk_score']:.0%}")
    print(f"Confidence : {result['confidence']:.0%}")
