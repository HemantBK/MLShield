# src/mlshield/detectors/layer3_llm.py
"""Layer 3: LLM Semantic Judge -- contextual analysis of complex security events."""
import httpx
import json
import os
from typing import Optional

from ..ingestion.event_bus import TrajectoryEvent
from ..specs.spec_types import ViolationResult


class LLMJudge:
    """
    Uses an LLM to provide semantic analysis of complex security events.

    Only called for <1% of events (Layer 2 medium-confidence zone).
    Provides natural language explanations and contextual threat analysis.
    """

    def __init__(
        self,
        api_url: str = "https://api.anthropic.com/v1/messages",
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

    async def judge(
        self,
        event: TrajectoryEvent,
        l1_result: ViolationResult,
        l2_score: float,
    ) -> dict:
        """
        Provide semantic judgment on a potentially malicious event.
        Only called for <1% of events (Layer 2 medium-confidence zone).
        """
        if not self.api_key:
            # No API key: fall back to heuristic
            return self._fallback_judge(event, l1_result, l2_score)

        prompt = self._build_prompt(event, l1_result, l2_score)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=10.0,
                )
                data = response.json()
                text = data["content"][0]["text"]
                return self._parse_response(text)

        except Exception:
            # Fallback: use Layer 2 score directly
            return self._fallback_judge(event, l1_result, l2_score)

    def _build_prompt(self, event, l1_result, l2_score: float) -> str:
        return f"""You are a security analyst for an ML training cluster.
Analyze this event and determine if it indicates a security threat.

EVENT:
- Action: {event.action}
- Resource: {event.resource}
- User: {event.user}
- Job ID: {event.job_id}
- Timestamp: {event.timestamp}
- Details: {json.dumps(event.details, default=str)}

CONTEXT:
- Layer 1 (rule engine) flagged: {l1_result.violation_type or 'no violation'}
- Layer 2 (ML detector) anomaly score: {l2_score:.3f}
- This is step {event.trajectory_step} in the job's trajectory

Respond with ONLY a JSON object:
{{
 "is_threat": true/false,
 "confidence": 0.0-1.0,
 "threat_type": "weight_exfiltration|data_poisoning|unauthorized_access|benign",
 "severity": "critical|high|medium|low",
 "description": "one-line summary",
 "explanation": "2-3 sentence reasoning"
}}"""

    def _parse_response(self, text: str) -> dict:
        try:
            clean = text.strip().strip("```json").strip("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return {
                "is_threat": False,
                "confidence": 0.5,
                "severity": "medium",
                "description": "Could not parse LLM response",
                "explanation": text[:200],
            }

    def _fallback_judge(
        self,
        event: TrajectoryEvent,
        l1_result: ViolationResult,
        l2_score: float,
    ) -> dict:
        """
        Heuristic fallback when LLM is unavailable.
        Uses Layer 1 + Layer 2 signals to produce a judgment.
        """
        is_threat = l2_score > 0.7

        # Boost confidence if Layer 1 also flagged
        if l1_result.is_violation:
            is_threat = True
            confidence = min(1.0, l2_score + 0.2)
        else:
            confidence = l2_score

        # Determine threat type from signals
        threat_type = "benign"
        severity = "low"
        description = "No threat detected"
        explanation = "Event appears normal based on heuristic analysis."

        if is_threat:
            action = event.action.lower()
            details = event.details or {}

            if "egress" in action or "network" in action:
                threat_type = "weight_exfiltration"
                severity = "critical"
                dest = details.get("destination", "unknown")
                description = f"Suspicious network egress to {dest}"
                explanation = (
                    f"Network egress detected with ML anomaly score {l2_score:.2f}. "
                    f"Destination {dest} may indicate data exfiltration. "
                    f"Layer 1 flagged: {l1_result.violation_type or 'none'}."
                )
            elif "exec" in action:
                threat_type = "unauthorized_access"
                severity = "high"
                cmd = details.get("command", "unknown")
                description = f"Suspicious exec: {cmd}"
                explanation = (
                    f"Pod exec with anomaly score {l2_score:.2f}. "
                    f"Command '{cmd}' executed on {event.resource}. "
                    f"This may indicate lateral movement or unauthorized access."
                )
            elif details.get("is_weight_access"):
                threat_type = "weight_exfiltration"
                severity = "critical"
                description = f"Unauthorized weight access: {event.resource}"
                explanation = (
                    f"Weight file access detected with anomaly score {l2_score:.2f}. "
                    f"Resource {event.resource} was accessed by {event.user or 'unknown'}. "
                    f"This may indicate model weight theft."
                )
            elif details.get("z_score", 0) > 3:
                threat_type = "unauthorized_access"
                severity = "high"
                metric = details.get("metric", "unknown")
                description = f"GPU anomaly: {metric} z-score {details.get('z_score', 0):.1f}"
                explanation = (
                    f"GPU metric {metric} deviated significantly from baseline. "
                    f"Combined with ML score {l2_score:.2f}, this suggests abnormal compute usage. "
                    f"Could indicate cryptojacking or unauthorized distillation."
                )
            else:
                threat_type = "unauthorized_access"
                severity = "medium"
                description = f"Anomalous activity: {event.action} on {event.resource}"
                explanation = (
                    f"Event scored {l2_score:.2f} by ML detector. "
                    f"Action {event.action} on {event.resource} is unusual for this job. "
                    f"Further investigation recommended."
                )

        return {
            "is_threat": is_threat,
            "confidence": confidence,
            "threat_type": threat_type,
            "severity": severity,
            "description": description,
            "explanation": explanation,
        }
