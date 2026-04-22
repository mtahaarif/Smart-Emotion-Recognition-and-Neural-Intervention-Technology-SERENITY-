import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional,Any
from datetime import datetime, timezone, timedelta


FRAMEWORK_DBT = "DBT_Distress_Tolerance"
FRAMEWORK_CBT = "CBT_Restructuring"
FRAMEWORK_ACT = "ACT_Defusion"
FRAMEWORK_SUPPORTIVE = "Supportive_Stabilization"

MODE_DBT = "DBT"
MODE_CBT = "CBT"
MODE_ACT = "ACT"
MODE_SUPPORTIVE = "SUPPORTIVE"

ACUTE_EMOTIONS = {"panic", "anger", "angry", "sad"}

ABSOLUTE_PATTERNS = [
    re.compile(r"\b(always|never|everyone|nobody)\b", re.IGNORECASE),
]

CATASTROPHIZING_PATTERNS = [
    re.compile(r"\b(ruined|disaster|catastrophe|catastrophic|worst|nothing will get better)\b", re.IGNORECASE),
]

RUMINATION_PATTERNS = [
    re.compile(r"\b(can't stop thinking about|cannot stop thinking about|can't stop thinking|cannot stop thinking|cant stop thinking about|cant stop thinking)\b", re.IGNORECASE),
    re.compile(r"\b(wish i hadn't|i wish i had not|wish i hadnt)\b", re.IGNORECASE),
]

ACUTE_DISTRESS_PATTERNS = [
    re.compile(r"\b(hopeless|worthless|overwhelmed|panic|can't cope|cannot cope)\b", re.IGNORECASE),
    re.compile(r"\b(self[-\s]?harm|suicid|hurt myself|end my life|kill myself|want to die|don't want to live)\b", re.IGNORECASE),
]


@dataclass
class RoutingDecision:
    framework: str
    route_reason: str
    route_locked: bool
    risk_score: int
    dominant_emotion: str
    speech_emotion: str
    face_emotion: str
    acute_safety_trigger: bool = False
    high_distress: bool = False
    rumination_detected: bool = False
    detected_distortions: List[str] = field(default_factory=list)


def _normalize_emotion(value: Optional[str]) -> str:
    return str(value or "neutral").strip().lower()


def _contains_any(patterns: List[re.Pattern], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def mode_to_framework(mode: str) -> str:
    normalized_mode = str(mode or MODE_SUPPORTIVE).strip().upper()
    if normalized_mode == MODE_DBT:
        return FRAMEWORK_DBT
    if normalized_mode == MODE_CBT:
        return FRAMEWORK_CBT
    if normalized_mode == MODE_ACT:
        return FRAMEWORK_ACT
    return FRAMEWORK_SUPPORTIVE


def determine_clinical_mode(user_text: str, risk_score: int, dominant_emotion: str) -> str:
    """Route each turn to a strict mode using lightweight NLP heuristics."""
    normalized_text = str(user_text or "").strip().lower()
    normalized_emotion = _normalize_emotion(dominant_emotion)

    acute_distress = _contains_any(ACUTE_DISTRESS_PATTERNS, normalized_text)
    if int(risk_score or 0) >= 7 or normalized_emotion in ACUTE_EMOTIONS or acute_distress:
        return MODE_DBT

    if _contains_any(ABSOLUTE_PATTERNS, normalized_text) or _contains_any(CATASTROPHIZING_PATTERNS, normalized_text):
        return MODE_CBT

    if _contains_any(RUMINATION_PATTERNS, normalized_text):
        return MODE_ACT

    return MODE_SUPPORTIVE


def detect_cognitive_distortions(text: str) -> List[str]:
    findings: List[str] = []
    if _contains_any(ABSOLUTE_PATTERNS, text):
        findings.append("absolutist_thinking")
    if _contains_any(CATASTROPHIZING_PATTERNS, text):
        findings.append("catastrophizing")
    return findings


def detect_rumination(text: str) -> bool:
    return _contains_any(RUMINATION_PATTERNS, text)


def detect_acute_safety_language(text: str) -> bool:
    return _contains_any(ACUTE_DISTRESS_PATTERNS[1:], text)


# Add/Update these sections in clinical_router.py

def evaluate_clinical_route(
    user_text: str,
    risk_score: int,
    dominant_emotion: str,
    speech_emotion: str,
    face_emotion: str,
    user_model: Any, # Pass the User DB object here
    forced_mode: Optional[str] = None,
) -> RoutingDecision:
    text = str(user_text or "").strip().lower()
    
    # 1. TARASOFF (Duty to Warn) HEURISTIC
    violence_pattern = r"(kill|hurt|stab|shoot|attack|make them pay)\s+(him|her|them|everyone|people)"
    if re.search(violence_pattern, text):
        user_model.duty_to_warn = True # Persists to DB
    
    # 2. COOL-DOWN INTERCEPTOR (24-Hour Rule)
    is_in_cooldown = False
    if user_model.last_crisis_timestamp:
        try:
            last_crisis = datetime.fromisoformat(user_model.last_crisis_timestamp)
            # Check if current time is within 24 hours of crisis
            if datetime.now(timezone.utc).replace(tzinfo=None) - last_crisis < timedelta(hours=24):
                is_in_cooldown = True
        except: pass

    # If in cooldown, force DBT (Stabilization) regardless of other triggers
    effective_mode = "DBT" if is_in_cooldown else (forced_mode or determine_clinical_mode(text, risk_score, dominant_emotion))
    
    normalized = text.lower()

    dom = _normalize_emotion(dominant_emotion)
    speech = _normalize_emotion(speech_emotion)
    face = _normalize_emotion(face_emotion)

    acute_safety = detect_acute_safety_language(normalized)
    mode = str(forced_mode or determine_clinical_mode(text, risk_score, dom)).strip().upper()

    high_distress = mode == MODE_DBT
    distortions = detect_cognitive_distortions(normalized)
    rumination = detect_rumination(normalized)

    if acute_safety:
        return RoutingDecision(
            framework=FRAMEWORK_DBT,
            route_reason="Acute self-harm language detected",
            route_locked=True,
            risk_score=int(risk_score),
            dominant_emotion=dom,
            speech_emotion=speech,
            face_emotion=face,
            acute_safety_trigger=True,
            high_distress=True,
            rumination_detected=rumination,
            detected_distortions=distortions,
        )

    if mode == MODE_DBT:
        return RoutingDecision(
            framework=FRAMEWORK_DBT,
            route_reason="Mode lock set to DBT from risk/emotion/distress heuristic",
            route_locked=True,
            risk_score=int(risk_score),
            dominant_emotion=dom,
            speech_emotion=speech,
            face_emotion=face,
            acute_safety_trigger=False,
            high_distress=True,
            rumination_detected=rumination,
            detected_distortions=distortions,
        )

    if mode == MODE_CBT:
        return RoutingDecision(
            framework=FRAMEWORK_CBT,
            route_reason="Mode lock set to CBT from absolutist language heuristic",
            route_locked=True,
            risk_score=int(risk_score),
            dominant_emotion=dom,
            speech_emotion=speech,
            face_emotion=face,
            acute_safety_trigger=False,
            high_distress=False,
            rumination_detected=rumination,
            detected_distortions=distortions,
        )

    if mode == MODE_ACT:
        return RoutingDecision(
            framework=FRAMEWORK_ACT,
            route_reason="Mode lock set to ACT from rumination heuristic",
            route_locked=True,
            risk_score=int(risk_score),
            dominant_emotion=dom,
            speech_emotion=speech,
            face_emotion=face,
            acute_safety_trigger=False,
            high_distress=False,
            rumination_detected=True,
            detected_distortions=distortions,
        )

    return RoutingDecision(
        framework=FRAMEWORK_SUPPORTIVE,
        route_reason="No strict framework trigger detected",
        route_locked=False,
        risk_score=int(risk_score),
        dominant_emotion=dom,
        speech_emotion=speech,
        face_emotion=face,
        acute_safety_trigger=False,
        high_distress=False,
        rumination_detected=False,
        detected_distortions=distortions,
    )


def build_routed_prompt(
    user_text: str,
    decision: RoutingDecision,
    clinical_phase: str,
    requires_safety_review: bool,
) -> str:
    safety_line = (
        "Client profile is flagged Requires_Safety_Review. Begin with one validating safety-oriented line and avoid minimization."
        if requires_safety_review
        else ""
    )

    framework_rules: Dict[str, str] = {
        FRAMEWORK_DBT: (
            "Use strict DBT mode only. Focus on distress tolerance and emotion regulation skills, including TIPP and STOP when appropriate. "
            "Prioritize immediate stabilization, grounding, paced breathing, and concrete next actions. "
            "Do NOT analyze automatic thoughts or perform cognitive restructuring in this mode."
        ),
        FRAMEWORK_CBT: (
            "Use strict CBT mode only. Identify the automatic thought, name the likely cognitive distortion, test evidence for and against, "
            "and guide the user to write one balanced alternative thought. Keep it structured and stepwise."
        ),
        FRAMEWORK_ACT: (
            "Use strict ACT mode only. Focus on cognitive defusion (help the user separate from thoughts), acceptance of internal experience, "
            "present-moment awareness, and one value-aligned committed action. Do NOT perform cognitive restructuring in this mode."
        ),
        FRAMEWORK_SUPPORTIVE: (
            "Use supportive, structured, non-diagnostic clinical coaching with reflective listening and practical coping steps."
        ),
    }

    framework_instruction = framework_rules.get(decision.framework, framework_rules[FRAMEWORK_SUPPORTIVE])
    distortion_hint = ", ".join(decision.detected_distortions) if decision.detected_distortions else "none"

    return (
        "You are SERENITY Clinical Agent. Respond with compassionate, non-diagnostic language. "
        f"Current routed framework: {decision.framework}. {framework_instruction} "
        f"Current protocol phase: {clinical_phase}. "
        f"Route reason: {decision.route_reason}. Detected distortions: {distortion_hint}. "
        f"{safety_line} "
        "Return a direct conversational response only; do not emit visible JSON in the chat text. "
        f"User input: {str(user_text or '').strip()}"
    )


def build_safety_override_response() -> Dict[str, object]:
    return {
        "response_text": (
            "I am switching to immediate safety support. Let's pause and stabilize together right now. "
            "Name 5 things you can see, 4 things you can feel, 3 things you can hear, 2 things you can smell, and 1 thing you can taste."
        ),
        "advance_phase": False,
        "detected_distortion": "",
        "safety_alert": True,
        "safety_reason": "Acute self-harm language or extreme distress detected",
    }

def determine_therapy_mode(user_text: str, user_profile: Any) -> tuple[str, bool]:
    """
    Returns (therapy_mode, duty_to_warn_flag)
    """
    # --- PROTOCOL 1: TARASOFF RULE (Duty to Warn) ---
    # Detect threats to OTHERS
    violence_pattern = r"(kill|hurt|stab|shoot|attack|make them pay)\s+(him|her|them|everyone|people)"
    is_threatening_others = bool(re.search(violence_pattern, user_text.lower()))
    
    if is_threatening_others:
        user_profile.duty_to_warn = True # This triggers the RED banner on Admin Observatory
        # We don't necessarily tell the user, but we log it for the clinician immediately.

    # --- PROTOCOL 2: POST-CRISIS COOL-DOWN (24-Hour Rule) ---
    if user_profile.last_crisis_timestamp:
        last_crisis = datetime.fromisoformat(user_profile.last_crisis_timestamp)
        # Check if 24 hours have passed
        if datetime.now(timezone.utc) - last_crisis < timedelta(hours=24):
            # STRICTURE: Force Stabilization mode. NO CBT, NO ACT allowed.
            # Clinically, the user needs "Psychological First Aid" right now.
            return "SUPPORTIVE_STABILIZATION", is_threatening_others

    # --- PROTOCOL 3: STANDARD SEMANTIC ROUTING ---
    # (Your existing keyword/LLM logic goes here)
    if "always" in user_text or "never" in user_text:
        return "CBT", is_threatening_others
    
    return "SUPPORTIVE", is_threatening_others
