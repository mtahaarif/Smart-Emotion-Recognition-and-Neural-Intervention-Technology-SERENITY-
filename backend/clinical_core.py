import re
from datetime import datetime
from typing import Any, Dict, List

WORD_RE = re.compile(r"[a-zA-Z']+")

OPENNESS_WORDS = {
    "curious",
    "explore",
    "learn",
    "reflect",
    "meaning",
    "creative",
    "insight",
    "growth",
    "journal",
    "understand",
}
CONSCIENTIOUSNESS_WORDS = {
    "plan",
    "routine",
    "schedule",
    "discipline",
    "organize",
    "goal",
    "complete",
    "consistent",
    "track",
    "focus",
}
EXTRAVERSION_WORDS = {
    "friend",
    "friends",
    "family",
    "social",
    "talk",
    "people",
    "meet",
    "team",
    "together",
    "group",
}
AGREEABLENESS_WORDS = {
    "thanks",
    "thank",
    "appreciate",
    "sorry",
    "care",
    "kind",
    "support",
    "help",
    "empathy",
    "understanding",
}
SELF_FOCUS_WORDS = {"i", "me", "my", "myself"}

TRAIT_INSIGHTS = {
    "openness": {
        "high": "Receptive to reflective exercises and values-based reframing.",
        "moderate": "Benefits from balanced structure plus guided exploration.",
        "low": "Responds better to concrete, practical coping steps.",
    },
    "conscientiousness": {
        "high": "Can usually follow structured routines and measurable goals.",
        "moderate": "Needs realistic, low-friction plans with reminders.",
        "low": "Works best with tiny daily habits and minimal cognitive load.",
    },
    "extraversion": {
        "high": "Social support and connection-based interventions are useful.",
        "moderate": "Alternates between social and solitary recovery strategies.",
        "low": "Prefers quiet, self-paced coping and reflective practices.",
    },
    "agreeableness": {
        "high": "Often responsive to collaborative and compassion-focused framing.",
        "moderate": "Responds to direct but warm communication.",
        "low": "Benefits from clear boundaries and pragmatic action plans.",
    },
    "neuroticism": {
        "high": "Needs stronger emotion-regulation and crisis-prevention scaffolding.",
        "moderate": "Benefits from regular emotional check-ins and coping rehearsals.",
        "low": "Can usually tolerate stress with lighter maintenance routines.",
    },
}


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in WORD_RE.findall(str(text or ""))]


def _bounded(score: float) -> float:
    return round(max(0.0, min(100.0, score)), 1)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _band(score: float) -> str:
    if score >= 67:
        return "high"
    if score <= 33:
        return "low"
    return "moderate"


def _trait_payload(key: str, label: str, score: float) -> Dict[str, Any]:
    band = _band(score)
    return {
        "key": key,
        "label": label,
        "score": _bounded(score),
        "band": band,
        "insight": TRAIT_INSIGHTS[key][band],
    }


def estimate_personality_profile(
    chats: List[Dict[str, Any]],
    latest_scores: Dict[str, int],
    distress_signal_count: int,
    negative_ratio: float,
    engagement_score: int,
) -> Dict[str, Any]:
    user_texts = [str(row.get("user_text") or "") for row in chats if str(row.get("user_text") or "").strip()]
    token_list = _tokenize(" ".join(user_texts))

    token_count = max(1, len(token_list))
    unique_ratio = _ratio(len(set(token_list)), token_count)
    avg_turn_len = _ratio(sum(len(_tokenize(txt)) for txt in user_texts), max(1, len(user_texts)))

    openness_hits = sum(1 for tok in token_list if tok in OPENNESS_WORDS)
    conscientious_hits = sum(1 for tok in token_list if tok in CONSCIENTIOUSNESS_WORDS)
    extraversion_hits = sum(1 for tok in token_list if tok in EXTRAVERSION_WORDS)
    agreeableness_hits = sum(1 for tok in token_list if tok in AGREEABLENESS_WORDS)
    self_focus_hits = sum(1 for tok in token_list if tok in SELF_FOCUS_WORDS)

    phq = int(latest_scores.get("PHQ-9", 0))
    gad = int(latest_scores.get("GAD-7", 0))
    pcl = int(latest_scores.get("PCL-5", 0))

    burden = (
        (_ratio(phq, 27) if phq else 0.0)
        + (_ratio(gad, 21) if gad else 0.0)
        + (_ratio(pcl, 80) if pcl else 0.0)
    ) / max(1, sum(1 for v in [phq, gad, pcl] if v > 0)) if any(v > 0 for v in [phq, gad, pcl]) else 0.0

    openness = 35 + (unique_ratio * 30) + (_ratio(openness_hits, token_count) * 220) + min(10, avg_turn_len * 0.8)
    conscientiousness = (
        32
        + (_ratio(conscientious_hits, token_count) * 230)
        + min(24, engagement_score * 0.22)
        + (6 if len(latest_scores) >= 2 else 0)
        - (burden * 12)
    )
    extraversion = 30 + (_ratio(extraversion_hits, token_count) * 240) + min(14, len(chats) * 0.25) + (avg_turn_len * 0.25)
    agreeableness = 34 + (_ratio(agreeableness_hits, token_count) * 260) + (_ratio(self_focus_hits, token_count) * -45)
    neuroticism = 24 + (negative_ratio * 44) + min(20, distress_signal_count * 4.5) + (burden * 28)

    traits = [
        _trait_payload("openness", "Openness", openness),
        _trait_payload("conscientiousness", "Conscientiousness", conscientiousness),
        _trait_payload("extraversion", "Extraversion", extraversion),
        _trait_payload("agreeableness", "Agreeableness", agreeableness),
        _trait_payload("neuroticism", "Neuroticism", neuroticism),
    ]

    ranked = sorted(traits, key=lambda row: row["score"], reverse=True)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "model": "heuristic_big5_v1",
        "confidence": "low",
        "disclaimer": "This is a non-diagnostic behavioral estimate based on interaction signals, not a clinical personality test.",
        "traits": traits,
        "dominant_traits": [row["label"] for row in ranked[:2]],
    }


def build_personalized_routine(
    risk_level: str,
    active_flags: List[str],
    dominant_emotion: str,
    latest_scores: Dict[str, int],
    screening_trends: Dict[str, str],
    distress_signal_count: int,
    negative_ratio: float,
    engagement_level: str,
    personality_profile: Dict[str, Any],
) -> Dict[str, Any]:
    trait_map = {
        row.get("key"): float(row.get("score", 50.0))
        for row in personality_profile.get("traits", [])
        if isinstance(row, dict)
    }

    neuroticism = trait_map.get("neuroticism", 50.0)
    conscientiousness = trait_map.get("conscientiousness", 50.0)
    extraversion = trait_map.get("extraversion", 50.0)
    openness = trait_map.get("openness", 50.0)

    has_depression_flag = "possible_depression" in set(active_flags)
    has_anxiety_flag = "possible_anxiety" in set(active_flags)
    has_trauma_flag = "possible_trauma_stress" in set(active_flags)

    focus_theme = "Stabilization and emotional safety"
    if risk_level == "stable":
        focus_theme = "Momentum building and resilience maintenance"
    elif risk_level == "monitor":
        focus_theme = "Symptom tracking and coping consistency"

    morning_actions = [
        "2-minute mood check-in (name emotion, intensity 0-10)",
        "Hydration + natural light exposure within first hour",
    ]
    daytime_actions = [
        "One focused work/study block using 25-minute intervals",
        "10-15 minute movement break",
    ]
    evening_actions = [
        "Brief reflection: what helped today, what was hard",
        "Wind-down routine with reduced screen stimulation before sleep",
    ]

    micro_interventions = [
        "When overwhelmed: 5-4-3-2-1 grounding and slow exhale",
        "When self-critical thoughts rise: evidence-for/evidence-against reframing",
        "When motivation is low: start with a 2-minute action rather than waiting for mood",
    ]

    if has_anxiety_flag or neuroticism >= 67:
        morning_actions.append("4-7-8 breathing for 3 cycles before major tasks")
        daytime_actions.append("Schedule a 10-minute worry window instead of all-day rumination")

    if has_depression_flag:
        morning_actions.append("Behavioral activation: one meaningful task before noon")
        evening_actions.append("Record one mastery moment and one pleasant moment")

    if has_trauma_flag:
        daytime_actions.append("Carry a grounding cue card for trigger moments")
        micro_interventions.append("If triggered: orient to present time/place before decision-making")

    if extraversion >= 60:
        daytime_actions.append("Social micro-dose: short check-in with a trusted person")
    else:
        evening_actions.append("Solo decompression block (quiet, low stimulation)")

    if openness >= 60:
        evening_actions.append("Reflective journaling prompt: what value guided me today?")

    if conscientiousness < 45:
        morning_actions = morning_actions[:2] + ["Pick one tiny non-negotiable habit only (keep it easy)"]

    if engagement_level == "low":
        micro_interventions.append("Set one phone reminder for a single daily check-in")

    weekly_targets = [
        {
            "title": "Track emotional intensity once daily",
            "metric": "At least 5 entries this week",
            "why": "Builds awareness and early detection of escalation patterns.",
        },
        {
            "title": "Complete one supportive activity",
            "metric": "3 activities this week",
            "why": "Behavioral activation improves mood inertia and self-efficacy.",
        },
    ]

    if negative_ratio >= 0.55:
        weekly_targets.append(
            {
                "title": "Thought-balance exercise",
                "metric": "At least 4 completed reframes this week",
                "why": "Reduces dominance of negative automatic thoughts.",
            }
        )

    next_screenings = []
    if latest_scores.get("PHQ-9", 0) >= 10 or screening_trends.get("PHQ-9") == "worsening":
        next_screenings.append("Repeat PHQ-9 in 7 days")
    if latest_scores.get("GAD-7", 0) >= 10 or screening_trends.get("GAD-7") == "worsening":
        next_screenings.append("Repeat GAD-7 in 7 days")
    if latest_scores.get("PCL-5", 0) >= 33 or screening_trends.get("PCL-5") == "worsening":
        next_screenings.append("Repeat PCL-5 in 10-14 days")
    if not next_screenings:
        next_screenings.append("Repeat selected questionnaire set in 2-4 weeks")

    safety_protocol = {
        "warning_signs": [
            "Rapid increase in hopelessness, panic, or shutdown",
            "Persistent sleep disruption for 3+ nights",
            "Repeated self-harm ideation language",
        ],
        "immediate_steps": [
            "Pause and do grounding + paced breathing for 2-5 minutes",
            "Contact a trusted person and state current distress clearly",
            "Reduce access to immediate self-harm means when possible",
        ],
        "escalation": "If there is imminent risk of self-harm, contact local emergency services immediately.",
        "distress_signal_count": int(distress_signal_count),
    }

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "focus_theme": focus_theme,
        "daily_routine": {
            "morning": morning_actions,
            "daytime": daytime_actions,
            "evening": evening_actions,
        },
        "micro_interventions": micro_interventions,
        "weekly_targets": weekly_targets,
        "monitoring": {
            "risk_level": str(risk_level or "stable"),
            "dominant_emotion": str(dominant_emotion or "neutral"),
            "next_screenings": next_screenings,
        },
        "safety_protocol": safety_protocol,
    }
