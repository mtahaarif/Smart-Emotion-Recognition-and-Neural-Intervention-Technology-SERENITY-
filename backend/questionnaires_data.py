from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Option sets
# ---------------------------------------------------------------------------
PHQ_GAD_OPTIONS = [
    {"value": 0, "label": "Not at all"},
    {"value": 1, "label": "Several days"},
    {"value": 2, "label": "More than half the days"},
    {"value": 3, "label": "Nearly every day"},
]
PCL5_OPTIONS = [
    {"value": 0, "label": "Not at all"},
    {"value": 1, "label": "A little bit"},
    {"value": 2, "label": "Moderately"},
    {"value": 3, "label": "Quite a bit"},
    {"value": 4, "label": "Extremely"},
]

# ---------------------------------------------------------------------------
# Questionnaire definitions
# ---------------------------------------------------------------------------
QUESTIONNAIRE_DEFINITIONS: Dict[str, dict] = {
    "PHQ-9": {
        "title": "PHQ-9 Depression Screen",
        "description": "Over the last two weeks, how often have you been bothered by the following?",
        "max_score": 27, "option_set": "PHQ_GAD",
        "questions": [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself or that you are a failure",
            "Trouble concentrating on things",
            "Moving or speaking so slowly that other people could notice",
            "Thoughts that you would be better off dead, or of hurting yourself",
        ],
    },
    "GAD-7": {
        "title": "GAD-7 Anxiety Screen",
        "description": "Over the last two weeks, how often have you been bothered by these concerns?",
        "max_score": 21, "option_set": "PHQ_GAD",
        "questions": [
            "Feeling nervous, anxious, or on edge",
            "Not being able to stop or control worrying",
            "Worrying too much about different things",
            "Trouble relaxing",
            "Being so restless that it is hard to sit still",
            "Becoming easily annoyed or irritable",
            "Feeling afraid as if something awful might happen",
        ],
    },
    "PCL-5": {
        "title": "PCL-5 Trauma Stress Screen",
        "description": "In the past month, how much were you bothered by the following?",
        "max_score": 80, "option_set": "PCL5",
        "questions": [
            "Repeated, disturbing memories of a stressful experience",
            "Repeated, disturbing dreams related to a stressful experience",
            "Suddenly feeling or acting as if the stressful experience were happening again",
            "Feeling very upset when something reminded you of the stressful experience",
            "Having strong physical reactions when reminded of the stressful experience",
            "Avoiding memories, thoughts, or feelings related to the stressful experience",
            "Avoiding external reminders of the stressful experience",
            "Trouble remembering important parts of the stressful experience",
            "Having strong negative beliefs about yourself, others, or the world",
            "Blaming yourself or someone else for the stressful experience",
            "Having strong negative feelings such as fear, anger, guilt, or shame",
            "Loss of interest in activities you used to enjoy",
            "Feeling distant or cut off from other people",
            "Trouble feeling positive emotions",
            "Irritable behavior, angry outbursts, or acting aggressively",
            "Taking too many risks or doing things that could cause harm",
            "Being super-alert, watchful, or on guard",
            "Feeling jumpy or easily startled",
            "Difficulty concentrating",
            "Trouble falling or staying asleep",
        ],
    },
}

# Canonical alias mapping
_ALIASES: Dict[str, str] = {}
for _k in QUESTIONNAIRE_DEFINITIONS:
    for _v in (_k, _k.lower(), _k.replace("-", ""), _k.replace("-", "_")):
        _ALIASES[_v] = _k


def normalize_questionnaire_type(value: str) -> Optional[str]:
    key = str(value or "").strip()
    return _ALIASES.get(key) or _ALIASES.get(key.lower())


# ---------------------------------------------------------------------------
# Severity tables — fast lookup
# ---------------------------------------------------------------------------
def severity_from_score(q_type: str, score: int) -> str:
    if q_type == "PHQ-9":
        if score <= 4:  return "minimal"
        if score <= 9:  return "mild"
        if score <= 14: return "moderate"
        if score <= 19: return "moderately severe"
        return "severe"
    if q_type == "GAD-7":
        if score <= 4:  return "minimal"
        if score <= 9:  return "mild"
        if score <= 14: return "moderate"
        return "severe"
    if q_type == "PCL-5":
        if score < 20: return "low"
        if score < 33: return "elevated"
        if score < 50: return "high"
        return "very high"
    return "unknown"


def score_questionnaire(q_type: str, answers: List[int]) -> Tuple[int, str]:
    defn     = QUESTIONNAIRE_DEFINITIONS[q_type]
    max_item = 4 if q_type == "PCL-5" else 3
    n        = len(defn["questions"])
    total    = sum(max(0, min(max_item, int(answers[i] if i < len(answers) else 0)))
                   for i in range(n))
    return total, severity_from_score(q_type, total)


# ---------------------------------------------------------------------------
# Clinical flags — thresholds from validated cut-points
# ---------------------------------------------------------------------------
def questionnaire_clinical_flags(latest_scores: Dict[str, int]) -> Dict[str, bool]:
    return {
        "possible_depression":    int(latest_scores.get("PHQ-9", 0)) >= 10,
        "possible_anxiety":       int(latest_scores.get("GAD-7", 0)) >= 10,
        "possible_trauma_stress": int(latest_scores.get("PCL-5", 0)) >= 33,
    }


# ---------------------------------------------------------------------------
# Template builder for frontend
# ---------------------------------------------------------------------------
def options_for_questionnaire(q_type: str) -> List[dict]:
    opt = QUESTIONNAIRE_DEFINITIONS[q_type].get("option_set", "PHQ_GAD")
    return [dict(o) for o in (PCL5_OPTIONS if opt == "PCL5" else PHQ_GAD_OPTIONS)]


def questionnaire_templates(selected_types: Optional[List[str]] = None) -> List[dict]:
    types = (
        list(QUESTIONNAIRE_DEFINITIONS) if not selected_types else
        [t for raw in selected_types if (t := normalize_questionnaire_type(raw))]
    )
    out = []
    for q_type in types:
        defn = QUESTIONNAIRE_DEFINITIONS.get(q_type)
        if not defn:
            continue
        out.append({
            "type":        q_type,
            "title":       defn["title"],
            "description": defn["description"],
            "max_score":   defn["max_score"],
            "options":     options_for_questionnaire(q_type),
            "questions":   [{"id": i + 1, "text": q}
                            for i, q in enumerate(defn["questions"])],
        })
    return out