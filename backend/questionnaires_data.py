from typing import Dict, List, Optional, Tuple

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

QUESTIONNAIRE_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "PHQ-9": {
        "type": "PHQ-9",
        "title": "PHQ-9 Depression Screen",
        "description": "Over the last two weeks, how often have you been bothered by the following problems?",
        "max_score": 27,
        "option_set": "PHQ_GAD",
        "questions": [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself, or that you are a failure, or have let yourself or your family down",
            "Trouble concentrating on things, such as reading or watching television",
            "Moving or speaking so slowly that other people could notice, or the opposite, being fidgety or restless",
            "Thoughts that you would be better off dead, or thoughts of hurting yourself",
        ],
    },
    "GAD-7": {
        "type": "GAD-7",
        "title": "GAD-7 Anxiety Screen",
        "description": "Over the last two weeks, how often have you been bothered by these concerns?",
        "max_score": 21,
        "option_set": "PHQ_GAD",
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
        "type": "PCL-5",
        "title": "PCL-5 Trauma Stress Screen",
        "description": "In the past month, how much were you bothered by the following problems?",
        "max_score": 80,
        "option_set": "PCL5",
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

ALIASES = {
    "phq9": "PHQ-9",
    "phq-9": "PHQ-9",
    "phq_9": "PHQ-9",
    "gad7": "GAD-7",
    "gad-7": "GAD-7",
    "gad_7": "GAD-7",
    "pcl5": "PCL-5",
    "pcl-5": "PCL-5",
    "pcl_5": "PCL-5",
}


def normalize_questionnaire_type(value: str) -> Optional[str]:
    key = str(value or "").strip()
    if not key:
        return None

    lowered = key.lower()
    if lowered in ALIASES:
        return ALIASES[lowered]

    for canonical in QUESTIONNAIRE_DEFINITIONS:
        if lowered == canonical.lower():
            return canonical

    return None


def options_for_questionnaire(questionnaire_type: str) -> List[Dict[str, object]]:
    option_set = str(QUESTIONNAIRE_DEFINITIONS[questionnaire_type].get("option_set", "PHQ_GAD"))
    if option_set == "PCL5":
        return [item.copy() for item in PCL5_OPTIONS]
    return [item.copy() for item in PHQ_GAD_OPTIONS]


def questionnaire_templates(selected_types: Optional[List[str]] = None) -> List[Dict[str, object]]:
    if not selected_types:
        selected = list(QUESTIONNAIRE_DEFINITIONS.keys())
    else:
        selected = []
        for value in selected_types:
            canonical = normalize_questionnaire_type(value)
            if canonical and canonical not in selected:
                selected.append(canonical)

    templates: List[Dict[str, object]] = []
    for questionnaire_type in selected:
        definition = QUESTIONNAIRE_DEFINITIONS.get(questionnaire_type)
        if definition is None:
            continue
        templates.append(
            {
                "type": questionnaire_type,
                "title": definition["title"],
                "description": definition["description"],
                "max_score": definition["max_score"],
                "options": options_for_questionnaire(questionnaire_type),
                "questions": [
                    {
                        "id": index + 1,
                        "text": question,
                    }
                    for index, question in enumerate(definition["questions"])
                ],
            }
        )

    return templates


def score_questionnaire(questionnaire_type: str, answers: List[int]) -> Tuple[int, str]:
    definition = QUESTIONNAIRE_DEFINITIONS[questionnaire_type]
    questions = definition["questions"]
    expected_count = len(questions)
    max_per_item = 4 if questionnaire_type == "PCL-5" else 3

    normalized_answers: List[int] = []
    for idx in range(expected_count):
        raw = answers[idx] if idx < len(answers) else 0
        try:
            score = int(raw)
        except Exception:
            score = 0
        score = max(0, min(max_per_item, score))
        normalized_answers.append(score)

    total_score = sum(normalized_answers)
    severity = severity_from_score(questionnaire_type, total_score)
    return total_score, severity


def severity_from_score(questionnaire_type: str, score: int) -> str:
    if questionnaire_type == "PHQ-9":
        if score <= 4:
            return "minimal"
        if score <= 9:
            return "mild"
        if score <= 14:
            return "moderate"
        if score <= 19:
            return "moderately severe"
        return "severe"

    if questionnaire_type == "GAD-7":
        if score <= 4:
            return "minimal"
        if score <= 9:
            return "mild"
        if score <= 14:
            return "moderate"
        return "severe"

    if questionnaire_type == "PCL-5":
        if score < 20:
            return "low"
        if score < 33:
            return "elevated"
        if score < 50:
            return "high"
        return "very high"

    return "unknown"


def questionnaire_clinical_flags(latest_scores: Dict[str, int]) -> Dict[str, bool]:
    return {
        "possible_depression": int(latest_scores.get("PHQ-9", 0)) >= 10,
        "possible_anxiety": int(latest_scores.get("GAD-7", 0)) >= 10,
        "possible_trauma_stress": int(latest_scores.get("PCL-5", 0)) >= 33,
    }
