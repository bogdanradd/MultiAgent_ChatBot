import json
import re
from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from src.config import ROUTER_MODEL, PROMPTS_DIR, OLLAMA_HOST

Intent = Literal["SUMMARIZE", "QA", "MCQ"]


class MCQ(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str


def classify(user_input: str) -> Intent:
    """Classify user intent using router LLM."""
    prompt_template = (PROMPTS_DIR / "router.txt").read_text()
    prompt = prompt_template.format(user_input=user_input)

    llm = ChatOllama(model=ROUTER_MODEL, temperature=0.0, base_url=OLLAMA_HOST)
    response = llm.invoke(prompt)

    classification = response.content.strip().upper()

    if classification in {"SUMMARIZE", "QA", "MCQ"}:
        return classification
    else:
        return "QA"


def handle(user_input: str, doc_id: str | None = None) -> dict:
    """
    Handle user input with CrewAI agentic workflow.

    Args:
        user_input: User's question/request
        doc_id: Optional document scope filter

    Returns:
        Dictionary with intent and result
    """
    from src.crew_agents import run_crew_workflow

    intent = classify(user_input)

    if intent == "SUMMARIZE":
        result = run_crew_workflow('summarize', doc_id=doc_id)
        result = str(result['result'])
    elif intent == "QA":
        result = run_crew_workflow('research', question=user_input, doc_id=doc_id)
        result = str(result['result'])
    elif intent == "MCQ":
        result = run_crew_workflow('insights', topic=user_input, doc_id=doc_id, n=5)
        result = _parse_crewai_mcqs(str(result['result']))

    return {"intent": intent, "result": result}


def _parse_crewai_mcqs(raw: str) -> list:
    """Parse CrewAI MCQ JSON output into MCQ objects."""
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not json_match:
        return raw

    try:
        items = json.loads(json_match.group())
        mcqs = []
        for item in items:
            options = item.get('options', [])[:4]
            if len(options) < 4:
                continue
            answer_index = item.get('correct_answer_index',
                           item.get('correctAnswerIndex',
                           item.get('answer_index', 0)))
            mcqs.append(MCQ(
                question=item.get('question', ''),
                options=options,
                answer_index=min(answer_index, 3),
                explanation=item.get('explanation', '')
            ))
        return mcqs if mcqs else raw
    except (json.JSONDecodeError, Exception):
        return raw
