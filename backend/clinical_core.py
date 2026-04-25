import json
import re
import textwrap
from collections import defaultdict
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Phase registry
# ---------------------------------------------------------------------------
PHASES_BY_FRAMEWORK: Dict[str, List[str]] = {
    "DBT_Distress_Tolerance":  [
        "Stabilization and Immediate Grounding", "Crisis Survival Skills",
        "Emotion Regulation During Peak Distress", "Post-Crisis Recovery Plan",
    ],
    "CBT_Restructuring": [
        "Identify Automatic Thoughts", "Label Cognitive Distortion",
        "Evidence Examination", "Balanced Reframe and Action",
    ],
    "ACT_Defusion": [
        "Notice Thought-Emotion Loop", "Defusion from Narrative",
        "Acceptance and Present-Moment Contact", "Values-Aligned Micro-Action",
    ],
    "Supportive_Stabilization": [
        "Emotional Check-In", "Clarify Needs and Stressors", "Coping Plan and Commitment",
    ],
}
_DEFAULT_FRAMEWORK = "Supportive_Stabilization"
_FALLBACK_PAYLOAD  = {"response_text": "", "advance_phase": False,
                      "detected_distortion": "", "safety_alert": False,
                      "safety_reason": "", "raw_payload": {}}
_JSON_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def default_phase_for_framework(framework: str) -> str:
    return PHASES_BY_FRAMEWORK.get(framework,
           PHASES_BY_FRAMEWORK[_DEFAULT_FRAMEWORK])[0]


def advance_phase(framework: str, current_phase: str) -> str:
    phases = PHASES_BY_FRAMEWORK.get(framework,
             PHASES_BY_FRAMEWORK[_DEFAULT_FRAMEWORK])
    try:
        return phases[min(phases.index(current_phase) + 1, len(phases) - 1)]
    except ValueError:
        return phases[0]


# ---------------------------------------------------------------------------
# LLM payload extractor — O(1) fast-fail regex, no multi-pass findall
# ---------------------------------------------------------------------------
def parse_structured_llm_payload(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {**_FALLBACK_PAYLOAD, "response_text": ""}

    def _extract(d: dict) -> Dict[str, Any]:
        resp = str(d.get("response_text") or d.get("response") or d.get("message") or "").strip()
        return {
            "response_text":       resp,
            "advance_phase":       bool(d.get("advance_phase", False)),
            "detected_distortion": str(d.get("detected_distortion") or "").strip(),
            "safety_alert":        bool(d.get("safety_alert", False)),
            "safety_reason":       str(d.get("safety_reason") or "").strip(),
            "raw_payload":         d,
        }

    # 1. Direct parse
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return _extract(d)
    except json.JSONDecodeError:
        pass

    # 2. Fast regex scan for first JSON object
    for m in _JSON_RE.finditer(text):
        try:
            d = json.loads(m.group())
            if isinstance(d, dict):
                return _extract(d)
        except json.JSONDecodeError:
            continue

    return {**_FALLBACK_PAYLOAD, "response_text": re.sub(r"\s+", " ", text)}


# ---------------------------------------------------------------------------
# Weekly trajectory — uses lightweight tuples instead of dict allocation
# ---------------------------------------------------------------------------
def compute_weekly_trajectory_flags(
    questionnaire_rows: List[Dict[str, Any]],
    worsening_delta: int = 4,
) -> Dict[str, Any]:
    since = datetime.utcnow() - timedelta(days=7)
    now   = datetime.utcnow()
    by_type: Dict[str, List[tuple]] = defaultdict(list)  # (datetime, score)

    for row in questionnaire_rows:
        q_type = str(row.get("questionnaire_type") or "").strip()
        if not q_type:
            continue
        try:
            created = datetime.fromisoformat(
                str(row.get("created_at") or "").replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except Exception:
            continue
        if created >= since:
            by_type[q_type].append((created, int(row.get("total_score") or 0)))

    snapshots, flagged = [], []
    for q_type, entries in by_type.items():
        if len(entries) < 2:
            continue
        entries.sort()                          # sort by datetime (first element)
        baseline, latest = entries[0][1], entries[-1][1]
        delta   = latest - baseline
        flagged_q = delta >= worsening_delta
        if flagged_q:
            flagged.append(q_type)
        snapshots.append({
            "questionnaire_type": q_type,
            "baseline_score": baseline, "latest_score": latest,
            "delta_score": delta, "window_days": 7,
            "flagged": flagged_q, "computed_at": now.isoformat(),
        })

    return {"snapshots": snapshots,
            "requires_safety_review": bool(flagged),
            "flagged_questionnaires": flagged}


# ---------------------------------------------------------------------------
# Markdown builders — minimal allocations
# ---------------------------------------------------------------------------
def build_handoff_markdown(
    username: str, risk_score: int, route_framework: str,
    active_flags: List[str], distress_signals: int,
    recent_turns: List[Dict[str, Any]],
) -> str:
    lines = [
        "# SERENITY Clinical Handoff Report", "",
        f"- Generated at: {datetime.utcnow().isoformat()} UTC",
        f"- Client: {username}",
        f"- Risk score: {int(risk_score)}",
        f"- Routed framework: {route_framework}",
        f"- Active flags: {', '.join(active_flags) if active_flags else 'none'}",
        f"- Distress signal count: {int(distress_signals)}", "",
        "## Recent Transcript Excerpts",
    ]
    if not recent_turns:
        lines.append("- No recent transcript turns available.")
    else:
        for row in recent_turns[:10]:
            lines += ["", f"### Turn {row.get('id')}",
                      f"- Timestamp: {row.get('timestamp')}",
                      f"- User: {str(row.get('user_text') or '')[:500]}",
                      f"- Assistant: {str(row.get('assistant_text') or '')[:500]}"]
    return "\n".join(lines).strip() + "\n"


def build_admin_handoff_markdown(
    user_id: int, username: str, risk_score: int,
    requires_safety_review: bool, active_framework: str,
    trajectory: Dict[str, Any], recent_turns: List[Dict[str, Any]],
    clinical_narrative: str = "", clinical_narrative_source: str = "",
) -> str:
    ls  = trajectory.get("latest_scores") or {}
    vd  = trajectory.get("velocity_delta") or {}
    fl  = trajectory.get("flagged_questionnaires") or []
    hist = trajectory.get("history") or {}

    lines: List[str] = ["# SERENITY Clinical Handoff Report", ""]
    if (narr := str(clinical_narrative or "").strip()):
        lines += ["## Clinical Handoff Narrative",
                  f"- Narrative source: {clinical_narrative_source or 'fallback'}",
                  "", narr, ""]

    lines += [
        "## Client Overview",
        f"- Generated at: {datetime.utcnow().isoformat()} UTC",
        f"- User ID: {user_id}  Username: {username}",
        f"- Active framework: {active_framework}",
        f"- Risk score: {risk_score}  Safety review: {'YES' if requires_safety_review else 'NO'}",
        "", "## Measurement-Based Care Snapshot",
        f"- Scores: {', '.join(f'{k}: {v}' for k,v in ls.items()) or 'unavailable'}",
        f"- Velocity: {', '.join(f'{k} delta: {v}' for k,v in vd.items()) or 'unavailable'}",
        f"- Flagged: {', '.join(fl) or 'none'}",
        "", "### Recent Trajectory Entries",
    ]
    for qt in ("PHQ-9", "GAD-7", "PCL-5"):
        rows = list(hist.get(qt) or [])
        if not rows:
            lines.append(f"- {qt}: no recent entries")
            continue
        compact = " | ".join(
            f"{r.get('created_at','?')} -> {r.get('score',0)} ({r.get('severity','?')})"
            for r in rows[-3:]
        )
        lines.append(f"- {qt}: {compact}")

    lines += ["", "## Recent Conversation Turns (Last 15)"]
    if not recent_turns:
        lines.append("- No turns available.")
    else:
        for row in recent_turns[:15]:
            lines += ["", f"### Turn {row.get('id')}",
                      f"- Timestamp: {row.get('timestamp','unknown')}",
                      f"- User: {str(row.get('user_text') or '')[:700]}",
                      f"- Assistant: {str(row.get('assistant_text') or '')[:700]}"]
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Admin prompt / fallback
# ---------------------------------------------------------------------------
def build_admin_clinical_handoff_prompt(
    snapshot: Dict[str, Any], recent_turns: List[Dict[str, Any]],
) -> str:
    risk      = snapshot.get("risk") or {}
    screening = snapshot.get("screening") or {}
    emotion   = snapshot.get("emotion") or {}
    follow_up = snapshot.get("follow_up") or {}
    scores    = (screening.get("latest_scores") or {})
    trends    = (screening.get("trends") or {})

    turns = [{"timestamp": str(r.get("timestamp") or ""),
              "user_text":  str(r.get("user_text") or "")[:500],
              "assistant_text": str(r.get("assistant_text") or "")[:500]}
             for r in list(recent_turns or [])[:10]]

    payload = {
        "username":             str(snapshot.get("username") or "unknown"),
        "risk_score":           int(risk.get("score") or 0),
        "risk_level":           str(risk.get("level") or "stable"),
        "active_flags":         list(risk.get("active_flags") or []),
        "distress_signal_count":int(risk.get("distress_signal_count") or 0),
        "dominant_emotion":     str(emotion.get("dominant_emotion") or "neutral"),
        "latest_scores":        {"PHQ-9": scores.get("PHQ-9"),
                                 "GAD-7": scores.get("GAD-7"),
                                 "PCL-5": scores.get("PCL-5")},
        "screening_trends":     {k: str(trends.get(k) or "insufficient_data")
                                 for k in ("PHQ-9", "GAD-7", "PCL-5")},
        "follow_up_priority":   str(follow_up.get("primary_priority") or ""),
        "recent_turns":         turns,
    }
    return (
        "You are an AI Clinical Administrative Assistant. Synthesize the following "
        "patient data into a structured 3-paragraph Clinical Handoff Report (SBAR) for a "
        "psychologist. No DSM-5 diagnoses. Concise Markdown with short headers.\n\n"
        f"Data:\n{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def build_admin_clinical_handoff_fallback(
    snapshot: Dict[str, Any], _recent_turns: List[Dict[str, Any]],
) -> str:
    risk      = snapshot.get("risk") or {}
    screening = snapshot.get("screening") or {}
    emotion   = snapshot.get("emotion") or {}
    follow_up = snapshot.get("follow_up") or {}
    trends    = screening.get("trends") or {}

    _TIER = {("severe",): "severe", ("critical",): "severe",
             ("elevated",): "high",  ("high",): "high",
             ("monitor",): "moderate", ("moderate",): "moderate"}
    score = int(risk.get("score") or 0)
    level = str(risk.get("level") or "stable").lower()
    tier  = next((v for k, v in _TIER.items() if level in k), None)
    if tier is None:
        tier = "severe" if score >= 8 else "high" if score >= 6 else "moderate" if score >= 3 else "low"

    _TIER_SENTENCE = {
        "severe":   "Strongly recommend immediate verification of safety plan adherence and escalation.",
        "high":     "Increase contact frequency and confirm the next safety-focused follow-up interval.",
        "moderate": "Reinforce coping practice and reassess risk at the next scheduled contact.",
        "low":      "Continue routine monitoring with supportive check-ins.",
    }
    _TREND = {"worsening": "worsening", "improving": "improving",
              "stable": "stable", "mixed": "mixed"}

    phq9_t = _TREND.get(str(trends.get("PHQ-9") or "").lower(), "insufficiently characterized")
    gad7_t = _TREND.get(str(trends.get("GAD-7") or "").lower(), "insufficiently characterized")
    affect = str(emotion.get("dominant_emotion") or "neutral").lower()
    focus  = str(follow_up.get("primary_priority") or "structured supportive follow-up").rstrip(".")
    for p in ("Continue ", "Maintain ", "Prioritize "):
        if focus.lower().startswith(p.lower()):
            focus = focus[len(p):]
            break
    focus = focus[:1].lower() + focus[1:] if focus else focus

    return (
        f"Patient presents with composite risk score {score} ({tier} concern). "
        f"Screening shows {phq9_t} depressive symptoms and {gad7_t} anxiety markers. "
        f"Dominant affect is {affect}. Immediate focus: {focus}. "
        f"{_TIER_SENTENCE[tier]}"
    )


# ---------------------------------------------------------------------------
# PDF renderer — lossless word-wrap (no data truncation)
# ---------------------------------------------------------------------------
def render_handoff_pdf(markdown_text: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise RuntimeError(f"PDF generation unavailable: {exc}") from exc

    buf = BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    W, H   = A4
    mx, my = 15 * mm, 15 * mm
    cy     = H - my
    LINE_H = 5.5 * mm

    for raw in str(markdown_text or "").splitlines():
        # Wrap long lines so no clinical data is silently lost
        for wrapped in textwrap.wrap(raw, width=110) or [" "]:
            if cy <= my:
                pdf.showPage()
                cy = H - my
            pdf.drawString(mx, cy, wrapped)
            cy -= LINE_H

    pdf.showPage()
    pdf.save()
    buf.seek(0)
    return buf.read()