import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Framework / Mode constants
# ---------------------------------------------------------------------------
FRAMEWORK_DBT       = "DBT_Distress_Tolerance"
FRAMEWORK_CBT       = "CBT_Restructuring"
FRAMEWORK_ACT       = "ACT_Defusion"
FRAMEWORK_SUPPORTIVE = "Supportive_Stabilization"

MODE_DBT        = "DBT"
MODE_CBT        = "CBT"
MODE_ACT        = "ACT"
MODE_SUPPORTIVE = "SUPPORTIVE"

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level, compile once)
# ---------------------------------------------------------------------------
_RE_FLAGS = re.IGNORECASE
_ABSOLUTE       = re.compile(r"\b(always|never|everyone|nobody)\b", _RE_FLAGS)
_CATASTROPHIZE  = re.compile(r"\b(ruined|disaster|catastrophe|catastrophic|worst|nothing will get better)\b", _RE_FLAGS)
_RUMINATE       = re.compile(r"\b(can'?t stop thinking (about)?|wish i had(n'?t)?|cannot stop thinking)\b", _RE_FLAGS)
_ACUTE_SAFETY   = re.compile(r"\b(self[-\s]?harm|suicid|hurt myself|end my life|kill myself|want to die|don'?t want to live)\b", _RE_FLAGS)
_ACUTE_DISTRESS = re.compile(r"\b(hopeless|worthless|overwhelmed|panic|can'?t cope|cannot cope)\b", _RE_FLAGS)
_VIOLENCE       = re.compile(r"(kill|hurt|stab|shoot|attack|make them pay)\s+(him|her|them|everyone|people)", _RE_FLAGS)

_ACUTE_EMOTIONS = {"panic", "anger", "angry", "sad"}


# ---------------------------------------------------------------------------
# Routing decision dataclass
# ---------------------------------------------------------------------------
@dataclass
class RoutingDecision:
    framework:           str
    route_reason:        str
    route_locked:        bool
    risk_score:          int
    dominant_emotion:    str
    speech_emotion:      str
    face_emotion:        str
    acute_safety_trigger: bool = False
    high_distress:       bool  = False
    rumination_detected: bool  = False
    detected_distortions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(v: Optional[str]) -> str:
    return str(v or "neutral").strip().lower()


def detect_cognitive_distortions(text: str) -> List[str]:
    out = []
    if _ABSOLUTE.search(text):     out.append("absolutist_thinking")
    if _CATASTROPHIZE.search(text): out.append("catastrophizing")
    return out


def detect_rumination(text: str) -> bool:
    return bool(_RUMINATE.search(text))


def detect_acute_safety_language(text: str) -> bool:
    return bool(_ACUTE_SAFETY.search(text))


def determine_clinical_mode(user_text: str, risk_score: int, dominant_emotion: str) -> str:
    t = str(user_text or "").lower()
    e = _norm(dominant_emotion)

    if risk_score >= 7 or e in _ACUTE_EMOTIONS or _ACUTE_DISTRESS.search(t) or _ACUTE_SAFETY.search(t):
        return MODE_DBT
    if _ABSOLUTE.search(t) or _CATASTROPHIZE.search(t):
        return MODE_CBT
    if _RUMINATE.search(t):
        return MODE_ACT
    return MODE_SUPPORTIVE


def mode_to_framework(mode: str) -> str:
    return {MODE_DBT: FRAMEWORK_DBT, MODE_CBT: FRAMEWORK_CBT,
            MODE_ACT: FRAMEWORK_ACT}.get(mode.upper(), FRAMEWORK_SUPPORTIVE)


# ---------------------------------------------------------------------------
# Core router — no hard-stops, all traffic reaches LLM
# ---------------------------------------------------------------------------
def evaluate_clinical_route(
    user_text: str,
    risk_score: int,
    dominant_emotion: str,
    speech_emotion: str,
    face_emotion: str,
    user_model: Any,
    forced_mode: Optional[str] = None,
) -> RoutingDecision:
    text = str(user_text or "").lower()
    dom  = _norm(dominant_emotion)
    spk  = _norm(speech_emotion)
    fce  = _norm(face_emotion)

    # Tarasoff heuristic — flag only, never block
    if user_model is not None and _VIOLENCE.search(text):
        try:
            user_model.duty_to_warn = True
        except Exception:
            pass

    # Post-crisis cool-down → force DBT stabilisation
    in_cooldown = False
    if user_model is not None:
        last_crisis = getattr(user_model, "last_crisis_timestamp", None)
        if last_crisis:
            try:
                lc = datetime.fromisoformat(str(last_crisis))
                if lc.tzinfo is None:
                    lc = lc.replace(tzinfo=timezone.utc)
                in_cooldown = (datetime.now(timezone.utc) - lc) < timedelta(hours=24)
            except Exception:
                pass

    mode = MODE_DBT if in_cooldown else str(forced_mode or determine_clinical_mode(text, risk_score, dom)).upper()

    acute_safety = detect_acute_safety_language(text)
    distortions  = detect_cognitive_distortions(text)
    rumination   = detect_rumination(text)

    # Map mode → framework
    framework = mode_to_framework(mode)

    return RoutingDecision(
        framework           = framework,
        route_reason        = (
            "Acute self-harm language"         if acute_safety  else
            f"Mode lock: {mode} from heuristics"
        ),
        route_locked        = mode != MODE_SUPPORTIVE,
        risk_score          = int(risk_score),
        dominant_emotion    = dom,
        speech_emotion      = spk,
        face_emotion        = fce,
        acute_safety_trigger= acute_safety,
        high_distress       = mode == MODE_DBT,
        rumination_detected = rumination,
        detected_distortions= distortions,
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
_FRAMEWORK_RULES: Dict[str, str] = {
    FRAMEWORK_DBT: (
        "Strict DBT mode. Focus on distress tolerance and emotion regulation (TIPP, STOP, TIPP). "
        "Prioritize grounding, paced breathing, and one concrete next action. "
        "Do NOT perform cognitive restructuring."
    ),
    FRAMEWORK_CBT: (
        "Strict CBT mode. Identify the automatic thought, name the cognitive distortion, "
        "test evidence for and against, guide toward one balanced reframe."
    ),
    FRAMEWORK_ACT: (
        "Strict ACT mode. Cognitive defusion, acceptance, present-moment awareness, "
        "and one value-aligned committed action. No cognitive restructuring."
    ),
    FRAMEWORK_SUPPORTIVE: (
        "Supportive, non-diagnostic clinical coaching with reflective listening and practical coping steps."
    ),
}


def build_routed_prompt(
    user_text: str,
    decision: RoutingDecision,
    clinical_phase: str,
    requires_safety_review: bool,
) -> str:
    safety_line = (
        "Client is flagged Requires_Safety_Review. Begin with one validating safety-oriented line."
        if requires_safety_review else ""
    )
    distortion_hint = ", ".join(decision.detected_distortions) if decision.detected_distortions else "none"
    rules = _FRAMEWORK_RULES.get(decision.framework, _FRAMEWORK_RULES[FRAMEWORK_SUPPORTIVE])

    return (
        "You are SERENITY Clinical Agent. Respond with compassionate, non-diagnostic language. "
        f"Framework: {decision.framework}. {rules} "
        f"Phase: {clinical_phase}. Route reason: {decision.route_reason}. "
        f"Distortions: {distortion_hint}. {safety_line} "
        "Return conversational text only; no visible JSON. "
        f"User: {str(user_text or '').strip()}"
    )


def build_safety_override_response() -> Dict[str, Any]:
    return {
        "response_text": (
            "I am switching to immediate safety support. Let's stabilize together right now. "
            "Name 5 things you can see, 4 you can feel, 3 you can hear, 2 you can smell, 1 you can taste."
        ),
        "advance_phase": False,
        "detected_distortion": "",
        "safety_alert": True,
        "safety_reason": "Acute self-harm language or extreme distress detected",
    }