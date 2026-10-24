from pathlib import Path
from ollama import Client
from src.llm_config import MODEL_NAME, OLLAMA_HOST
from src.schemas import ModelAnswer


PROMPT_PATH = Path("prompts/system_prompt.txt")


def load_system_prompt():
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def create_client(api_key):
    return Client(host=OLLAMA_HOST, headers={"Authorization": "Bearer " + api_key})


def ask_llm(client, question_id, question, context):
    system_prompt = load_system_prompt()

    prompt = (
        f"{system_prompt}\n\n"
        f"question_id: {question_id}\n"
        f"Вопрос: {question}\n\n"
        f"Контекст:\n{context}"
    )

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format=ModelAnswer.model_json_schema()
    )

    return ModelAnswer.model_validate_json(response.message.content)


def validate_answer(answer, allowed_ids):
    errors = []

    used_ids = (answer.evidence_ids + answer.counterevidence_ids)

    for claim_id in used_ids:
        if claim_id not in allowed_ids:
            errors.append(f"Неизвестный claim_id: {claim_id}")

    if answer.status == "both":
        if not answer.evidence_ids:
            errors.append("Для both нет evidence_ids")

        if not answer.counterevidence_ids:
            errors.append("Для both нет counterevidence_ids")

    if answer.status == "insufficient":
        if answer.answer is not None:
            errors.append("Для insufficient answer должен быть null")

    return errors