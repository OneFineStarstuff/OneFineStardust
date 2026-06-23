"""
Global Systemic Risk Index (G-SRI) Calculator
Formalizes the systemic risk assessment for G-SIFIs mandated by Omni-Sentinel Governance.
"""

from typing import Dict, List, Optional

class GSRICalculator:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "containment_risk": 0.4,
            "integrity_violation": 0.3,
            "drift_deviation": 0.2,
            "ethical_non_compliance": 0.1
        }
        self.threshold = 40.0

    def calculate(self, metrics: Dict[str, float]) -> float:
        """
        Calculates the G-SRI score based on provided metrics.
        Higher score indicates higher systemic risk.
        """
        score = 0.0
        for key, weight in self.weights.items():
            val = metrics.get(key, 0.0)
            score += val * weight * 10 # Normalizing factor

        return round(score, 2)

    def get_status(self, score: float) -> str:
        if score >= self.threshold:
            return "CRITICAL (RED)"
        if score >= self.threshold * 0.75:
            return "WARNING (AMBER)"
        return "NOMINAL (GREEN)"

if __name__ == "__main__":
    # Example calculation matching Linear Issue AXI-13
    calc = GSRICalculator()
    sample_metrics = {
        "containment_risk": 1.2,
        "integrity_violation": 0.5,
        "drift_deviation": 0.3,
        "ethical_non_compliance": 0.2
    }
    score = calc.calculate(sample_metrics)
    print(f"Calculated G-SRI: {score}")
    print(f"Status: {calc.get_status(score)}")
