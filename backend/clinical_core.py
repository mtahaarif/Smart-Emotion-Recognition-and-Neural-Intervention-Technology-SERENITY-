import re
from datetime import datetime
from typing import Any, Dict, List

WORD_RE = re.compile(r"[a-zA-Z']+")

DISTORTION_RULES = [
    {
        "key": "all_or_nothing",
        "label": "All-or-Nothing Thinking",
        "pattern": re.compile(r"\b(always|never|completely|totally|nothing|everything)\b", re.IGNORECASE),
        "challenge": "What is a more balanced view between extremes?",
    },
    {
        "key": "catastrophizing",
        "label": "Catastrophizing",
        "pattern": re.compile(r"\b(disaster|ruined|unbearable|worst|can't survive|cannot survive|hopeless)\b", re.IGNORECASE),
        "challenge": "What is the most likely outcome, and how would I cope if it happens?",
    },
    {
        "key": "mind_reading",
        "label": "Mind Reading",
        "pattern": re.compile(r"\b(they think|everyone thinks|he thinks|she thinks|they hate me|they will judge me)\b", re.IGNORECASE),
        "challenge": "What direct evidence do I have for what others are thinking?",
    },
    {
        "key": "fortune_telling",
        "label": "Fortune Telling",
        "pattern": re.compile(r"\b(i will fail|it's going to fail|it will go wrong|nothing will work)\b", re.IGNORECASE),
        "challenge": "What are three other possible outcomes besides the worst-case one?",
    },
    {
        "key": "overgeneralization",
        "label": "Overgeneralization",
        "pattern": re.compile(r"\b(every time|nothing ever|no one ever|everyone always|nobody)\b", re.IGNORECASE),
        "challenge": "Can I find exceptions that show this is not true in every case?",
    },
    {
        "key": "should_statements",
        "label": "Should Statements",
        "pattern": re.compile(r"\b(should|must|have to|ought to)\b", re.IGNORECASE),
        "challenge": "What would a kinder, more flexible statement sound like?",
    },
    {
        "key": "labeling",
        "label": "Global Labeling",
        "pattern": re.compile(r"\b(i am a failure|i am useless|i am broken|i am worthless|loser)\b", re.IGNORECASE),
        "challenge": "Am I defining my whole identity by one moment or one setback?",
    },
    {
        "key": "personalization",
        "label": "Personalization",
        "pattern": re.compile(r"\b(it's all my fault|it is all my fault|i caused everything|i ruin everything)\b", re.IGNORECASE),
        "challenge": "Which factors were outside my control in this situation?",
    },
]

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


def detect_cognitive_distortions(thought_text: str) -> List[Dict[str, Any]]:
    text = str(thought_text or "").strip()
    if not text:
        return []

    findings: List[Dict[str, Any]] = []
    for rule in DISTORTION_RULES:
        match = rule["pattern"].search(text)
        if not match:
            continue

        findings.append(
            {
                "key": rule["key"],
                "label": rule["label"],
                "evidence": match.group(0),
                "challenge_prompt": rule["challenge"],
            }
        )

    return findings


def build_cbt_guided_prompts(
    risk_level: str,
    dominant_emotion: str,
    latest_scores: Dict[str, int],
    negative_ratio: float,
) -> Dict[str, Any]:
    risk = str(risk_level or "stable").lower()
    emotion = str(dominant_emotion or "neutral").lower()

    opening = [
        "Describe one specific moment from today that felt emotionally intense.",
        "What automatic thought showed up in that moment? Quote it exactly.",
        "Rate how strong the emotion felt from 0 to 10.",
    ]

    if emotion in {"sad", "fear", "angry", "disgust"}:
        opening.append("Where do you feel this in your body, and what urge comes with it?")

    evidence_prompts = [
        "What facts support this thought?",
        "What facts do not fully support this thought?",
        "If a close friend had this thought, what would you tell them?",
    ]

    reframe_prompts = [
        "Write one balanced alternative thought that is realistic and compassionate.",
        "Rate your emotion again from 0 to 10 after reframing.",
        "Choose one small action you can take in the next 24 hours.",
    ]

    safety_reminder = "If you feel at immediate risk of self-harm, contact local emergency services right away."
    if risk == "elevated":
        safety_reminder = (
            "High-risk flag detected: prioritize grounding, contact a trusted person now, and seek urgent local support if safety worsens."
        )

    focus_hint = "Build cognitive flexibility and reduce automatic negative thought dominance."
    if latest_scores.get("GAD-7", 0) >= 10:
        focus_hint = "Focus on probability-based thinking and uncertainty tolerance."
    elif latest_scores.get("PHQ-9", 0) >= 10:
        focus_hint = "Focus on hopelessness reframing and behavioral activation."
    elif latest_scores.get("PCL-5", 0) >= 33:
        focus_hint = "Focus on trigger grounding and present-moment orientation before reframing."

    if negative_ratio >= 0.55:
        reframe_prompts.append("Add one evidence log entry daily to interrupt negative spiral momentum.")

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "focus_hint": focus_hint,
        "opening_prompts": opening,
        "evidence_prompts": evidence_prompts,
        "reframe_prompts": reframe_prompts,
        "safety_reminder": safety_reminder,
        "session_goal": "Create one complete thought record with a measurable intensity reduction.",
    }


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def forecast_relapse_risk(
    profile: Dict[str, Any],
    clinical_parameters: Dict[str, Any],
    cbt_progress: Dict[str, Any],
    checkin_summary: Dict[str, Any],
) -> Dict[str, Any]:
    risk_score = int(profile.get("risk_score", clinical_parameters.get("risk_score", 0)) or 0)
    distress_signal_count = int(
        profile.get("distress_signal_count", clinical_parameters.get("distress_signal_count", 0)) or 0
    )
    negative_ratio = float(
        profile.get("negative_emotion_ratio", clinical_parameters.get("negative_emotion_ratio", 0.0)) or 0.0
    )
    symptom_burden_pct = float(
        profile.get("symptom_burden_pct", clinical_parameters.get("symptom_burden_pct", 0.0)) or 0.0
    )
    overall_trend = str(profile.get("overall_trend", clinical_parameters.get("overall_trend", "insufficient_data")) or "insufficient_data")
    engagement_level = str(profile.get("engagement_level", clinical_parameters.get("engagement_level", "low")) or "low").lower()

    cbt_trend = str(cbt_progress.get("trend") or "insufficient_data").lower()
    cbt_improvement_pct = float(cbt_progress.get("improvement_pct") or 0.0)
    cbt_completion_rate = float(cbt_progress.get("completion_rate") or 0.0)
    cbt_streak_days = int(cbt_progress.get("streak_days") or 0)
    cbt_total_records = int(cbt_progress.get("total_records") or 0)

    checkin_count = int(checkin_summary.get("count") or 0)
    avg_mood = float(checkin_summary.get("avg_mood") or 0.0)
    avg_stress = float(checkin_summary.get("avg_stress") or 0.0)
    avg_sleep = float(checkin_summary.get("avg_sleep_hours") or 0.0)

    raw_points = 0.0
    contributors: List[Dict[str, Any]] = []

    def add(delta: float, reason: str) -> None:
        nonlocal raw_points
        raw_points += delta
        contributors.append({"driver": reason, "impact": round(delta, 2)})

    add(risk_score * 5.0, f"Current risk score contribution ({risk_score})")
    add((symptom_burden_pct / 100.0) * 18.0, f"Symptom burden contribution ({round(symptom_burden_pct, 1)}%)")
    add(min(12.0, distress_signal_count * 2.0), f"Distress language events ({distress_signal_count})")
    add(max(0.0, negative_ratio) * 16.0, f"Negative affect ratio ({round(negative_ratio, 3)})")

    if overall_trend == "worsening":
        add(8.0, "Screening trend worsening")
    elif overall_trend == "mixed":
        add(4.0, "Screening trend mixed")
    elif overall_trend == "improving":
        add(-4.0, "Screening trend improving")

    if cbt_trend == "worsening":
        add(9.0, "CBT distress trend worsening")
    elif cbt_trend == "stable":
        add(2.0, "CBT distress trend stable")
    elif cbt_trend == "improving":
        add(-8.0, "CBT distress trend improving")
    else:
        add(3.0, "CBT trend uncertain due to limited records")

    if cbt_improvement_pct >= 20.0:
        add(-6.0, f"Strong CBT intensity reduction ({round(cbt_improvement_pct, 1)}%)")
    elif cbt_improvement_pct <= 5.0:
        add(4.0, f"Minimal CBT intensity reduction ({round(cbt_improvement_pct, 1)}%)")

    if cbt_completion_rate >= 80.0:
        add(-3.0, f"High CBT completion quality ({round(cbt_completion_rate, 1)}%)")
    elif cbt_completion_rate < 50.0:
        add(4.0, f"Low CBT completion quality ({round(cbt_completion_rate, 1)}%)")

    if cbt_streak_days >= 3:
        add(-3.0, f"Protective CBT streak ({cbt_streak_days} days)")
    elif cbt_total_records > 0 and cbt_streak_days == 0:
        add(3.0, "Interrupted CBT practice streak")

    if checkin_count >= 3:
        if avg_stress >= 7.0:
            add(8.0, f"Sustained high stress in check-ins ({round(avg_stress, 1)}/10)")
        elif avg_stress <= 4.0:
            add(-2.0, f"Lower stress trend in check-ins ({round(avg_stress, 1)}/10)")

        if avg_mood <= 4.0:
            add(8.0, f"Low mood trend in check-ins ({round(avg_mood, 1)}/10)")
        elif avg_mood >= 7.0:
            add(-4.0, f"Higher mood trend in check-ins ({round(avg_mood, 1)}/10)")

        if avg_sleep < 6.0:
            add(6.0, f"Short sleep trend ({round(avg_sleep, 1)}h)")
        elif avg_sleep >= 7.0:
            add(-3.0, f"Protective sleep duration trend ({round(avg_sleep, 1)}h)")
    else:
        add(3.0, "Limited check-in coverage for stability forecasting")

    if engagement_level == "low":
        add(4.0, "Low engagement level")
    elif engagement_level == "high":
        add(-3.0, "High engagement level")

    relapse_probability_pct = int(max(0.0, min(100.0, round(raw_points))))

    if relapse_probability_pct >= 75:
        band = "critical"
    elif relapse_probability_pct >= 55:
        band = "high"
    elif relapse_probability_pct >= 35:
        band = "moderate"
    else:
        band = "low"

    coverage_score = 0
    if cbt_total_records >= 3:
        coverage_score += 1
    if checkin_count >= 4:
        coverage_score += 1
    if isinstance(profile.get("latest_scores"), dict) and len(profile.get("latest_scores") or {}) > 0:
        coverage_score += 1
    if int(profile.get("engagement_score") or 0) >= 20:
        coverage_score += 1

    confidence = "high" if coverage_score >= 3 else "moderate" if coverage_score == 2 else "low"

    warning_signs: List[str] = []
    if distress_signal_count > 0:
        warning_signs.append("Distress language detected in recent interactions")
    if overall_trend in {"worsening", "mixed"}:
        warning_signs.append(f"Screening trajectory is {overall_trend}")
    if cbt_trend in {"worsening", "stable"} and cbt_total_records > 0:
        warning_signs.append(f"CBT trend is {cbt_trend} with incomplete relief")
    if checkin_count >= 3 and avg_sleep < 6.0:
        warning_signs.append("Sleep duration is below protective threshold")
    if checkin_count >= 3 and avg_stress >= 7.0:
        warning_signs.append("Check-ins show sustained high stress")
    if checkin_count < 3:
        warning_signs.append("Insufficient recent check-ins for robust stability monitoring")

    protective_signals: List[str] = []
    if cbt_trend == "improving":
        protective_signals.append("CBT intensity trend is improving")
    if cbt_completion_rate >= 70.0:
        protective_signals.append("CBT record completion quality is strong")
    if cbt_streak_days >= 3:
        protective_signals.append("Consistent CBT streak maintained")
    if checkin_count >= 3 and avg_mood >= 6.5:
        protective_signals.append("Mood trend in check-ins is generally stable to positive")
    if checkin_count >= 3 and avg_sleep >= 7.0:
        protective_signals.append("Sleep trend remains in protective range")
    if engagement_level in {"moderate", "high"}:
        protective_signals.append("Engagement level indicates continued treatment participation")

    preventive_actions: List[str] = []
    if band in {"critical", "high"}:
        preventive_actions.extend([
            "Schedule clinician follow-up within 24-72 hours and reassess safety plan.",
            "Increase daily monitoring: mood/stress/sleep check-in plus one brief coping log.",
            "Prioritize trigger management and means-restriction conversation if safety concern escalates.",
        ])
    elif band == "moderate":
        preventive_actions.extend([
            "Maintain weekly clinician review focused on trigger-response patterns.",
            "Complete at least 4 CBT thought records this week with full evidence sections.",
            "Reinforce sleep and behavioral activation as relapse-prevention anchors.",
        ])
    else:
        preventive_actions.extend([
            "Continue maintenance plan with weekly CBT practice and routine check-ins.",
            "Use early warning checklist proactively when distress starts to rise.",
            "Review resilience wins each week to preserve protective momentum.",
        ])

    if checkin_count < 3:
        preventive_actions.append("Submit at least 4 check-ins this week to improve relapse visibility.")
    if cbt_completion_rate < 60.0:
        preventive_actions.append("Complete balanced thought plus action step for each CBT record to increase intervention quality.")

    ranked_contributors = sorted(contributors, key=lambda row: abs(float(row.get("impact") or 0.0)), reverse=True)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "relapse_probability_pct": relapse_probability_pct,
        "band": band,
        "confidence": confidence,
        "contributors": ranked_contributors[:8],
        "warning_signs": _dedupe_keep_order(warning_signs),
        "protective_signals": _dedupe_keep_order(protective_signals),
        "preventive_actions": _dedupe_keep_order(preventive_actions),
    }


def _screening_rows(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    latest_scores = profile.get("latest_scores") if isinstance(profile.get("latest_scores"), dict) else {}
    latest_severity = profile.get("latest_severity") if isinstance(profile.get("latest_severity"), dict) else {}
    screening_trends = profile.get("screening_trends") if isinstance(profile.get("screening_trends"), dict) else {}

    rows: List[Dict[str, Any]] = []
    for name, score in latest_scores.items():
        rows.append(
            {
                "type": str(name),
                "score": int(score or 0),
                "severity": str(latest_severity.get(name) or "unknown"),
                "trend": str(screening_trends.get(name) or "insufficient_data"),
            }
        )
    return rows


def build_clinician_handoff_report(
    username: str,
    overview: Dict[str, Any],
    cbt_progress: Dict[str, Any],
    recent_cbt_records: List[Dict[str, Any]],
    checkin_summary: Dict[str, Any],
    relapse_forecast: Dict[str, Any],
) -> Dict[str, Any]:
    profile = overview.get("profile") if isinstance(overview.get("profile"), dict) else {}
    clinical = overview.get("clinical_parameters") if isinstance(overview.get("clinical_parameters"), dict) else {}
    summary_text = str(overview.get("summary") or "").strip()

    screening_rows = _screening_rows(profile)
    risk_level = str(profile.get("risk_level") or clinical.get("risk_level") or "stable")
    risk_score = int(profile.get("risk_score", clinical.get("risk_score", 0)) or 0)
    relapse_probability = int(relapse_forecast.get("relapse_probability_pct") or 0)
    relapse_band = str(relapse_forecast.get("band") or "low")

    top_distortions = [
        str(row.get("distortion") or "").replace("_", " ").strip()
        for row in (cbt_progress.get("top_distortions") or [])
        if str(row.get("distortion") or "").strip()
    ]
    top_distortions = [item for item in top_distortions if item]

    cbt_snapshot = {
        "records_last_window": int(cbt_progress.get("total_records") or 0),
        "trend": str(cbt_progress.get("trend") or "insufficient_data"),
        "improvement_pct": float(cbt_progress.get("improvement_pct") or 0.0),
        "completion_rate": float(cbt_progress.get("completion_rate") or 0.0),
        "streak_days": int(cbt_progress.get("streak_days") or 0),
        "top_distortions": top_distortions[:5],
    }

    adherence_snapshot = {
        "checkin_count": int(checkin_summary.get("count") or 0),
        "avg_mood": float(checkin_summary.get("avg_mood") or 0.0),
        "avg_stress": float(checkin_summary.get("avg_stress") or 0.0),
        "avg_energy": float(checkin_summary.get("avg_energy") or 0.0),
        "avg_sleep_hours": float(checkin_summary.get("avg_sleep_hours") or 0.0),
    }

    handoff_priorities: List[str] = []
    if relapse_band in {"critical", "high"}:
        handoff_priorities.append("Immediate review of safety status and escalation thresholds.")
    if str(profile.get("overall_trend") or "") == "worsening":
        handoff_priorities.append("Prioritize worsening screening trajectory in next encounter.")
    if cbt_snapshot["completion_rate"] < 60.0:
        handoff_priorities.append("Coach completion quality for thought records (balanced thought + action plan).")
    if adherence_snapshot["avg_sleep_hours"] and adherence_snapshot["avg_sleep_hours"] < 6.0:
        handoff_priorities.append("Target sleep stabilization as a high-yield relapse prevention lever.")
    if not handoff_priorities:
        handoff_priorities.append("Maintain current prevention strategy and monitor for early warning shifts.")

    next_7_day_plan = (relapse_forecast.get("preventive_actions") or [])[:6]
    escalation_criteria = [
        "Rapid escalation in hopelessness or self-harm ideation language.",
        "Marked increase in screening severity or relapse probability into critical range.",
        "Sustained sleep collapse (<5h) with rising stress and reduced functioning.",
        "Inability to maintain basic safety plan commitments.",
    ]

    summary_bullets = [
        line.replace("-", "", 1).strip()
        for line in summary_text.splitlines()
        if line.strip()
    ]
    if not summary_bullets and summary_text:
        summary_bullets = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", summary_text) if seg.strip()]
    summary_bullets = summary_bullets[:6]

    markdown_lines = [
        f"# Clinical Handoff: {username}",
        "",
        f"Generated: {datetime.utcnow().isoformat()}",
        f"Current risk level: {risk_level} (score {risk_score})",
        f"Relapse forecast: {relapse_probability}% ({relapse_band})",
        "",
        "## Summary",
    ]
    if summary_bullets:
        markdown_lines.extend([f"- {line}" for line in summary_bullets])
    else:
        markdown_lines.append("- No summary available.")

    markdown_lines.extend([
        "",
        "## Screening Snapshot",
    ])
    if screening_rows:
        for row in screening_rows:
            markdown_lines.append(
                f"- {row['type']}: score {row['score']} ({row['severity']}), trend {row['trend']}"
            )
    else:
        markdown_lines.append("- No recent screening data.")

    markdown_lines.extend([
        "",
        "## CBT Snapshot",
        f"- Records in window: {cbt_snapshot['records_last_window']}",
        f"- Trend: {cbt_snapshot['trend']}",
        f"- Improvement: {cbt_snapshot['improvement_pct']}%",
        f"- Completion quality: {cbt_snapshot['completion_rate']}%",
        f"- Streak: {cbt_snapshot['streak_days']} day(s)",
    ])
    if cbt_snapshot["top_distortions"]:
        markdown_lines.append(f"- Top distortions: {', '.join(cbt_snapshot['top_distortions'])}")

    markdown_lines.extend([
        "",
        "## Handoff Priorities",
    ])
    markdown_lines.extend([f"- {item}" for item in handoff_priorities])

    markdown_lines.extend([
        "",
        "## Next 7-Day Plan",
    ])
    markdown_lines.extend([f"- {item}" for item in next_7_day_plan])

    markdown_lines.extend([
        "",
        "## Escalation Criteria",
    ])
    markdown_lines.extend([f"- {item}" for item in escalation_criteria])

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "username": username,
        "triage": {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "relapse_probability_pct": relapse_probability,
            "relapse_band": relapse_band,
        },
        "summary_bullets": summary_bullets,
        "screening_snapshot": screening_rows,
        "cbt_snapshot": cbt_snapshot,
        "adherence_snapshot": adherence_snapshot,
        "warning_signs": relapse_forecast.get("warning_signs") or [],
        "protective_signals": relapse_forecast.get("protective_signals") or [],
        "handoff_priorities": handoff_priorities,
        "next_7_day_plan": next_7_day_plan,
        "escalation_criteria": escalation_criteria,
        "recent_cbt_records": recent_cbt_records[:5],
        "report_markdown": "\n".join(markdown_lines),
    }
