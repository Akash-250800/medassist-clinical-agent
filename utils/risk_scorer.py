"""
Risk Scorer Utility
XAI-inspired risk scoring for clinical decision support
Parses LLM output and computes structured risk scores
"""

import json
import re
from typing import Dict, List, Optional


class RiskScorer:
    """
    Parses risk assessment output from LLM and computes numeric risk scores.

    Inspired by Explainable AI (XAI) principles - each risk factor
    contributes a weighted score for transparency.
    """

    # Risk factor weights (evidence-based)
    RISK_WEIGHTS = {
        # Red flag keywords → high weight
        "chest pain": 0.35,
        "crushing": 0.40,
        "radiating": 0.30,
        "shortness of breath": 0.25,
        "loss of consciousness": 0.45,
        "sudden onset": 0.35,
        "stroke": 0.50,
        "facial drooping": 0.45,
        "arm weakness": 0.40,
        "thunderclap headache": 0.45,
        "severe bleeding": 0.45,
        "anaphylaxis": 0.50,
        "sepsis": 0.50,
        "altered consciousness": 0.45,
        "oxygen saturation": 0.35,

        # Moderate risk keywords
        "fever": 0.15,
        "hypertension": 0.20,
        "diabetes": 0.15,
        "tachycardia": 0.20,
        "elevated troponin": 0.40,
        "drug interaction": 0.25,
        "renal failure": 0.30,
        "heart failure": 0.30,

        # Low risk
        "fatigue": 0.05,
        "mild pain": 0.05,
        "nausea": 0.10,
        "dizziness": 0.12,
    }

    URGENCY_SCORES = {
        "EMERGENCY": 0.90,
        "URGENT": 0.60,
        "ROUTINE": 0.20
    }

    def parse_and_score(self, llm_output: str) -> Dict:
        """
        Parse LLM risk assessment output and compute risk score.

        Args:
            llm_output: JSON string from risk assessment node

        Returns:
            Dict with red_flags, risk_factors, drug_warnings, urgency,
                      risk_score, confidence, score_breakdown
        """
        # Try to parse JSON from LLM output
        parsed = self._extract_json(llm_output)

        if not parsed:
            # Fallback: extract from text
            parsed = self._extract_from_text(llm_output)

        # Compute risk score
        risk_score, score_breakdown = self._compute_risk_score(
            red_flags=parsed.get("red_flags", []),
            risk_factors=parsed.get("risk_factors", []),
            urgency=parsed.get("urgency", "ROUTINE")
        )

        parsed["risk_score"] = round(min(risk_score, 1.0), 3)
        parsed["score_breakdown"] = score_breakdown
        parsed["confidence"] = parsed.get("confidence", 0.75)

        return parsed

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON block from LLM response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in text
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _extract_from_text(self, text: str) -> Dict:
        """Fallback: extract risk info from unstructured text."""
        text_lower = text.lower()

        red_flags = []
        risk_factors = []
        drug_warnings = []

        # Extract red flags
        red_flag_section = re.search(
            r'red flags?[:\s]+(.*?)(?=risk factors?|drug|$)',
            text_lower, re.DOTALL
        )
        if red_flag_section:
            flags_text = red_flag_section.group(1)
            red_flags = [f.strip() for f in re.split(r'[-•\n,]', flags_text) if f.strip()][:5]

        # Detect urgency
        urgency = "ROUTINE"
        if "emergency" in text_lower:
            urgency = "EMERGENCY"
        elif "urgent" in text_lower:
            urgency = "URGENT"

        # Detect drug warnings
        if "drug interaction" in text_lower or "contraindicated" in text_lower:
            drug_warnings = ["Potential drug interaction detected - review medications"]

        return {
            "red_flags": red_flags,
            "risk_factors": risk_factors,
            "drug_warnings": drug_warnings,
            "urgency": urgency,
            "confidence": 0.60
        }

    def _compute_risk_score(
        self,
        red_flags: List[str],
        risk_factors: List[str],
        urgency: str
    ) -> tuple:
        """
        Compute weighted risk score from extracted factors.
        XAI principle: each factor's contribution is logged for transparency.

        Returns:
            (total_score, score_breakdown dict)
        """
        score_breakdown = {}
        total_score = 0.0

        # Score from urgency level
        urgency_score = self.URGENCY_SCORES.get(urgency.upper(), 0.2)
        score_breakdown["urgency_level"] = urgency_score
        total_score += urgency_score * 0.4  # Urgency = 40% weight

        # Score from red flags
        flag_score = 0.0
        for flag in red_flags:
            flag_lower = flag.lower()
            for keyword, weight in self.RISK_WEIGHTS.items():
                if keyword in flag_lower:
                    flag_score = max(flag_score, weight)
                    score_breakdown[f"flag_{keyword}"] = weight

        total_score += flag_score * 0.45  # Red flags = 45% weight

        # Score from risk factors
        factor_score = 0.0
        for factor in risk_factors:
            factor_lower = factor.lower()
            for keyword, weight in self.RISK_WEIGHTS.items():
                if keyword in factor_lower:
                    factor_score = max(factor_score, weight * 0.5)  # Lower weight for risk factors
                    score_breakdown[f"factor_{keyword}"] = weight * 0.5

        total_score += factor_score * 0.15  # Risk factors = 15% weight

        return total_score, score_breakdown

    def get_risk_label(self, score: float) -> str:
        """Convert numeric score to human-readable label."""
        if score >= 0.75:
            return "🔴 HIGH RISK"
        elif score >= 0.45:
            return "🟡 MODERATE RISK"
        else:
            return "🟢 LOW RISK"

    def get_score_explanation(self, score_breakdown: Dict) -> str:
        """
        Generate XAI-style explanation of the risk score.
        Shows which factors contributed most to the score.
        """
        if not score_breakdown:
            return "Risk score computed from clinical pattern matching."

        # Sort by contribution
        sorted_factors = sorted(
            score_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 contributors

        explanation = "Risk Score Contributors:\n"
        for factor, contribution in sorted_factors:
            factor_name = factor.replace("flag_", "🚩 ").replace("factor_", "⚠️ ").replace("_", " ").title()
            bar = "█" * int(contribution * 10)
            explanation += f"  {factor_name}: {bar} ({contribution:.0%})\n"

        return explanation
