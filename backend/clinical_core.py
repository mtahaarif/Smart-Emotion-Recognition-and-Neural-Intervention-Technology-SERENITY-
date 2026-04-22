import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional


PHASES_BY_FRAMEWORK: Dict[str, List[str]] = {
    "DBT_Distress_Tolerance": [
        "Stabilization and Immediate Grounding",
        "Crisis Survival Skills",
        "Emotion Regulation During Peak Distress",
        "Post-Crisis Recovery Plan",
    ],
    "CBT_Restructuring": [
        "Identify Automatic Thoughts",
        "Label Cognitive Distortion",
        "Evidence Examination",
        "Balanced Reframe and Action",
    ],
    "ACT_Defusion": [
        "Notice Thought-Emotion Loop",
        "Defusion from Narrative",
        "Acceptance and Present-Moment Contact",
        "Values-Aligned Micro-Action",
    ],
    "Supportive_Stabilization": [
        "Emotional Check-In",
        "Clarify Needs and Stressors",
        "Coping Plan and Commitment",
    ],
}

DEFAULT_WEEKLY_WORSENING_DELTA = 4


def default_phase_for_framework(framework: str) -> str:
    phases = PHASES_BY_FRAMEWORK.get(framework) or PHASES_BY_FRAMEWORK["Supportive_Stabilization"]
    return phases[0]


def advance_phase(framework: str, current_phase: str) -> str:
    phases = PHASES_BY_FRAMEWORK.get(framework) or PHASES_BY_FRAMEWORK["Supportive_Stabilization"]
    if current_phase not in phases:
        return phases[0]
    idx = phases.index(current_phase)
    return phases[min(idx + 1, len(phases) - 1)]


def parse_structured_llm_payload(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()

    def _fallback() -> Dict[str, Any]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return {
            "response_text": cleaned,
            "advance_phase": False,
            "detected_distortion": "",
            "safety_alert": False,
            "safety_reason": "",
        }

    if not text:
        return _fallback()

    candidates: List[str] = [text]

    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)

    brace_matches = re.findall(r"(\{(?:.|\n)*\})", text)
    candidates.extend(brace_matches)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        response_text = str(payload.get("response_text") or payload.get("response") or "").strip()
        if not response_text and "message" in payload:
            response_text = str(payload.get("message") or "").strip()

        return {
            "response_text": response_text,
            "advance_phase": bool(payload.get("advance_phase", False)),
            "detected_distortion": str(payload.get("detected_distortion") or "").strip(),
            "safety_alert": bool(payload.get("safety_alert", False)),
            "safety_reason": str(payload.get("safety_reason") or "").strip(),
            "raw_payload": payload,
        }

    return _fallback()


def compute_weekly_trajectory_flags(
    questionnaire_rows: List[Dict[str, Any]],
    worsening_delta: int = DEFAULT_WEEKLY_WORSENING_DELTA,
) -> Dict[str, Any]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    now = datetime.utcnow()
    since = now - timedelta(days=7)

    for row in questionnaire_rows:
        q_type = str(row.get("questionnaire_type") or "").strip()
        if not q_type:
            continue
        created_raw = row.get("created_at")
        score = int(row.get("total_score") or 0)
        try:
            created = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        if created >= since:
            by_type[q_type].append({"score": score, "created_at": created})

    snapshots: List[Dict[str, Any]] = []
    flagged_types: List[str] = []

    for q_type, entries in by_type.items():
        if len(entries) < 2:
            continue
        ordered = sorted(entries, key=lambda item: item["created_at"])
        baseline = int(ordered[0]["score"])
        latest = int(ordered[-1]["score"])
        delta = latest - baseline
        flagged = delta >= int(worsening_delta)
        if flagged:
            flagged_types.append(q_type)
        snapshots.append(
            {
                "questionnaire_type": q_type,
                "baseline_score": baseline,
                "latest_score": latest,
                "delta_score": delta,
                "window_days": 7,
                "flagged": flagged,
                "computed_at": now.isoformat(),
            }
        )

    return {
        "snapshots": snapshots,
        "requires_safety_review": len(flagged_types) > 0,
        "flagged_questionnaires": flagged_types,
    }


def build_handoff_markdown(
    username: str,
    risk_score: int,
    route_framework: str,
    active_flags: List[str],
    distress_signals: int,
    recent_turns: List[Dict[str, Any]],
) -> str:
    lines = [
        "# SERENITY Clinical Handoff Report",
        "",
        f"- Generated at: {datetime.utcnow().isoformat()} UTC",
        f"- Client: {username}",
        f"- Risk score: {int(risk_score)}",
        f"- Routed framework: {route_framework}",
        f"- Active flags: {', '.join(active_flags) if active_flags else 'none'}",
        f"- Distress signal count: {int(distress_signals)}",
        "",
        "## Recent Transcript Excerpts",
    ]

    if not recent_turns:
        lines.append("- No recent transcript turns available.")
    else:
        for row in recent_turns[:10]:
            user_text = str(row.get("user_text") or "").strip().replace("\n", " ")
            assistant_text = str(row.get("assistant_text") or "").strip().replace("\n", " ")
            lines.extend(
                [
                    "",
                    f"### Turn {row.get('id')}",
                    f"- Timestamp: {row.get('timestamp')}",
                    f"- User: {user_text[:500]}",
                    f"- Assistant: {assistant_text[:500]}",
                ]
            )

    return "\n".join(lines).strip() + "\n"


def build_admin_handoff_markdown(
    user_id: int,
    username: str,
    risk_score: int,
    requires_safety_review: bool,
    active_framework: str,
    trajectory: Dict[str, Any],
    recent_turns: List[Dict[str, Any]],
    clinical_narrative: str = "",
    clinical_narrative_source: str = "",
) -> str:
    generated_at = datetime.utcnow().isoformat()
    latest_scores = trajectory.get("latest_scores") or {}
    velocity_delta = trajectory.get("velocity_delta") or {}
    flagged = trajectory.get("flagged_questionnaires") or []
    history = trajectory.get("history") or {}

    lines: List[str] = [
        "# SERENITY Clinical Handoff Report",
        "",
    ]

    narrative_text = str(clinical_narrative or "").strip()
    if narrative_text:
        lines.extend(
            [
                "## Clinical Handoff Narrative",
                f"- Narrative source: {str(clinical_narrative_source or 'fallback').strip()}",
                "",
                narrative_text,
                "",
            ]
        )

    lines.extend(
        [
        "## Client Overview",
        f"- Generated at: {generated_at} UTC",
        f"- User ID: {int(user_id)}",
        f"- Username: {username}",
        f"- Active framework: {active_framework}",
        f"- Latest risk score: {int(risk_score)}",
        f"- Requires safety review: {'YES' if bool(requires_safety_review) else 'NO'}",
        "",
        "## Measurement-Based Care Snapshot",
        ]
    )

    if latest_scores:
        score_line = ", ".join(f"{k}: {v}" for k, v in latest_scores.items())
        lines.append(f"- Latest questionnaire scores: {score_line}")
    else:
        lines.append("- Latest questionnaire scores: unavailable")

    if velocity_delta:
        velocity_line = ", ".join(
            f"{k} delta: {v if v is not None else 'insufficient_data'}"
            for k, v in velocity_delta.items()
        )
        lines.append(f"- Velocity markers: {velocity_line}")
    else:
        lines.append("- Velocity markers: unavailable")

    lines.append(
        f"- Flagged questionnaires: {', '.join(flagged) if flagged else 'none'}"
    )

    lines.extend(["", "### Recent Trajectory Entries"])
    for questionnaire_type in ("PHQ-9", "GAD-7", "PCL-5"):
        rows = list(history.get(questionnaire_type) or [])
        if not rows:
            lines.append(f"- {questionnaire_type}: no recent entries")
            continue

        compact_rows = []
        for row in rows[-3:]:
            created = str(row.get("created_at") or "unknown")
            score = int(row.get("score") or 0)
            severity = str(row.get("severity") or "unknown")
            compact_rows.append(f"{created} -> {score} ({severity})")
        lines.append(f"- {questionnaire_type}: {' | '.join(compact_rows)}")

    lines.extend(["", "## Recent Conversation Turns (Last 15)"])
    if not recent_turns:
        lines.append("- No conversation turns available.")
    else:
        for row in recent_turns[:15]:
            turn_id = row.get("id")
            timestamp = str(row.get("timestamp") or "unknown")
            user_text = str(row.get("user_text") or "").strip().replace("\n", " ")
            assistant_text = str(row.get("assistant_text") or "").strip().replace("\n", " ")
            lines.extend(
                [
                    "",
                    f"### Turn {turn_id}",
                    f"- Timestamp: {timestamp}",
                    f"- User: {user_text[:700]}",
                    f"- Assistant: {assistant_text[:700]}",
                ]
            )

    return "\n".join(lines).strip() + "\n"


def build_admin_clinical_handoff_prompt(
    snapshot: Dict[str, Any],
    recent_turns: List[Dict[str, Any]],
) -> str:
    risk = dict(snapshot.get("risk") or {})
    screening = dict(snapshot.get("screening") or {})
    emotion = dict(snapshot.get("emotion") or {})
    follow_up = dict(snapshot.get("follow_up") or {})

    latest_scores = dict(screening.get("latest_scores") or {})
    trends = dict(screening.get("trends") or {})

    compact_turns: List[Dict[str, str]] = []
    for row in list(recent_turns or [])[:10]:
        compact_turns.append(
            {
                "timestamp": str(row.get("timestamp") or "unknown"),
                "user_text": str(row.get("user_text") or "").strip()[:500],
                "assistant_text": str(row.get("assistant_text") or "").strip()[:500],
            }
        )

    payload = {
        "username": str(snapshot.get("username") or "unknown"),
        "risk_score": int(risk.get("score") or 0),
        "risk_level": str(risk.get("level") or "stable"),
        "active_flags": list(risk.get("active_flags") or []),
        "distress_signal_count": int(risk.get("distress_signal_count") or 0),
        "negative_affect_ratio": float(emotion.get("negative_ratio") or 0.0),
        "dominant_emotion": str(emotion.get("dominant_emotion") or "neutral"),
        "latest_scores": {
            "PHQ-9": latest_scores.get("PHQ-9"),
            "GAD-7": latest_scores.get("GAD-7"),
            "PCL-5": latest_scores.get("PCL-5"),
        },
        "screening_trends": {
            "PHQ-9": str(trends.get("PHQ-9") or "insufficient_data"),
            "GAD-7": str(trends.get("GAD-7") or "insufficient_data"),
            "PCL-5": str(trends.get("PCL-5") or "insufficient_data"),
            "overall": str(screening.get("overall_trend") or "insufficient_data"),
        },
        "follow_up_priority": str(follow_up.get("primary_priority") or ""),
        "monitoring_cadence": str(follow_up.get("cadence") or ""),
        "last_10_conversation_turns": compact_turns,
    }

    compact_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        "You are an AI Clinical Administrative Assistant. "
        "Your job is to synthesize the following raw patient data into a structured 3-paragraph Clinical Handoff Report for a human psychologist. "
        "Use the SBAR format (Situation, Background, Assessment, Recommendation). "
        "Do NOT make definitive DSM-5 diagnoses. Focus on observable behaviors, distress signals, and screening trends. "
        "Return Markdown with short section headers and concise clinical language.\n\n"
        f"Raw patient data:\n{compact_json}"
    )


def build_admin_clinical_handoff_fallback(
    snapshot: Dict[str, Any],
    recent_turns: List[Dict[str, Any]],
) -> str:
    risk = dict(snapshot.get("risk") or {})
    screening = dict(snapshot.get("screening") or {})
    emotion = dict(snapshot.get("emotion") or {})
    follow_up = dict(snapshot.get("follow_up") or {})

    latest_scores = dict(screening.get("latest_scores") or {})
    trends = dict(screening.get("trends") or {})

    def _trend_phrase(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized == "worsening":
            return "worsening"
        if normalized == "improving":
            return "improving"
        if normalized == "stable":
            return "stable"
        if normalized == "mixed":
            return "mixed"
        return "insufficiently characterized"

    def _risk_tier(score: int, level: str) -> str:
        normalized_level = str(level or "").strip().lower()
        if normalized_level in {"severe", "critical"} or score >= 8:
            return "severe"
        if normalized_level in {"elevated", "high"} or score >= 6:
            return "high"
        if normalized_level in {"monitor", "moderate"} or score >= 3:
            return "moderate"
        return "low"

    def _tier_sentence(tier: str) -> str:
        if tier == "severe":
            return "Strongly recommend immediate verification of safety plan adherence and escalation to primary psychiatric care provider."
        if tier == "high":
            return "Increase contact frequency, review protective factors, and confirm the next safety-focused follow-up interval."
        if tier == "moderate":
            return "Reinforce coping practice, monitor symptom drift, and reassess risk at the next scheduled contact."
        return "Continue routine monitoring with brief supportive check-ins and repeat screening at the next scheduled interval."

    risk_score = int(risk.get("score") or 0)
    risk_level = str(risk.get("level") or "stable")
    risk_tier = _risk_tier(risk_score, risk_level)
    follow_up_focus = str(follow_up.get("primary_priority") or "continue structured supportive follow-up").strip().rstrip(".")
    for prefix in ("Continue ", "Maintain ", "Prioritize "):
        if follow_up_focus.lower().startswith(prefix.lower()):
            follow_up_focus = follow_up_focus[len(prefix):]
            break
    if follow_up_focus and follow_up_focus[:1].isupper():
        follow_up_focus = follow_up_focus[:1].lower() + follow_up_focus[1:]
    dominant_affect = str(emotion.get("dominant_emotion") or "neutral").strip().lower() or "neutral"
    phq9_trend = _trend_phrase(trends.get("PHQ-9") or "insufficient_data")
    gad7_trend = _trend_phrase(trends.get("GAD-7") or "insufficient_data")

    narrative = (
        f"Patient presents with a composite risk score of {risk_score}, indicating a {risk_tier} level of concern. "
        f"Recent screening trajectories show {phq9_trend} depressive symptoms and {gad7_trend} anxiety markers. "
        f"Dominant interaction affect is {dominant_affect}. "
        f"Immediate clinical focus should prioritize {follow_up_focus}."
    )

    return f"{narrative} {_tier_sentence(risk_tier)}"


def render_handoff_pdf(markdown_text: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"PDF generation unavailable: {exc}") from exc

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin_x = 15 * mm
    margin_y = 15 * mm
    cursor_y = height - margin_y

    for raw_line in str(markdown_text or "").splitlines():
        line = raw_line if raw_line else " "
        if cursor_y <= margin_y:
            pdf.showPage()
            cursor_y = height - margin_y
        pdf.drawString(margin_x, cursor_y, line[:120])
        cursor_y -= 6 * mm

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.read()
