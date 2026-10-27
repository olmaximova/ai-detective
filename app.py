import json
import os
import random
import time

import pandas as pd
import streamlit as st

from src.context_builder import (
    build_flat_context,
    build_full_context,
    build_padic_context,
)
from src.evaluation import (
    compare_flat_padic,
    evaluate_answer,
    evaluate_stress_test,
)
from src.llm import (
    ask_llm,
    create_client,
    validate_answer,
)
from src.retrieval import get_top_claims

st.set_page_config(
    page_title="AI-детектив",
    layout="wide",
)

st.title("AI-детектив")


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_uploaded_jsonl(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8")
    rows = []

    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))

    return rows


def copy_rows(rows):
    return json.loads(
        json.dumps(
            rows,
            ensure_ascii=False,
        )
    )


def remove_claim(
    claim_id,
    documents,
    claims,
):
    new_documents = copy_rows(documents)
    new_claims = copy_rows(claims)

    claim = next(item for item in new_claims if item["claim_id"] == claim_id)

    for document in new_documents:
        if document["document_id"] == claim["document_id"]:
            document["text"] = document["text"].replace(
                claim["evidence"],
                "[УДАЛЕНО]",
            )

    new_claims = [item for item in new_claims if item["claim_id"] != claim_id]

    return new_documents, new_claims


CONTRADICTIONS = {
    "Q01": "Экспортный архив был создан не в 18:16, а в 18:19.",
    "Q02": "Резервный ключ R-17 выдала не Ольга Максимова.",
    "Q03": "Технический экспорт после 18:00 не был разрешён правилами.",
    "Q04": "Публичный архив не совпадал с внутренней сборкой команды.",
    "Q05": "Передача в /public-share/temp произошла раньше создания архива.",
    "Q06": "Резервный ключ R-17 был выдан после создания архива.",
    "Q07": "Технический экспорт в 18:16 нарушал правило заморозки.",
    "Q08": "Резервный ключ R-17 разрешал публичную публикацию сборки.",
    "Q09": "В 18:15 Вера точно не находилась рядом с WS-03.",
    "Q10": "Диана прямо попросила Веру публично опубликовать сборку.",
    "Q11": "Вера намеренно организовала утечку сборки.",
    "Q12": "Копирование в 18:18 лично выполнила Вера Соколова.",
}


DELETION_QUESTIONS = {
    "Q03",
    "Q04",
    "Q05",
    "Q06",
}


def add_contradiction(
    question_id,
    documents,
    claims,
):
    new_documents = copy_rows(documents)
    new_claims = copy_rows(claims)

    stress_id = f"S_{question_id}"
    text = CONTRADICTIONS[question_id]

    stress_document = {
        "document_id": "D_STRESS",
        "title": "Подготовленное противоречие",
        "source_type": "statement",
        "created_at": "2026-03-14T18:50:00",
        "author": "Дополнительный свидетель",
        "text": text,
    }

    stress_claim = {
        "claim_id": stress_id,
        "text": text,
        "document_id": "D_STRESS",
        "evidence": text,
        "status": "disputed",
    }

    new_documents.append(stress_document)
    new_claims.append(stress_claim)

    return (
        new_documents,
        new_claims,
        stress_id,
    )


def get_top_claims_with_stress(
    question,
    claims,
    stress_claim_id=None,
):
    top_claims = get_top_claims(
        question["question"],
        claims,
        top_k=15,
    )

    if stress_claim_id is None:
        return top_claims

    top_ids = {claim["claim_id"] for claim in top_claims}

    if stress_claim_id in top_ids:
        return top_claims

    stress_claim = next(
        claim for claim in claims if claim["claim_id"] == stress_claim_id
    )

    if len(top_claims) == 15:
        top_claims[-1] = stress_claim
    else:
        top_claims.append(stress_claim)

    return top_claims


def build_context(
    mode,
    question,
    documents,
    claims,
    stress_claim_id=None,
):
    documents_by_id = {document["document_id"]: document for document in documents}

    top_claims = get_top_claims_with_stress(
        question,
        claims,
        stress_claim_id,
    )

    if mode == "A":
        context = build_full_context(
            documents,
            claims,
        )

        allowed_ids = {claim["claim_id"] for claim in claims}

    elif mode == "B":
        context = build_flat_context(top_claims)

        allowed_ids = {claim["claim_id"] for claim in top_claims}

    else:
        context = build_padic_context(
            top_claims,
            documents_by_id,
        )

        allowed_ids = {claim["claim_id"] for claim in top_claims}

    return (
        context,
        top_claims,
        allowed_ids,
    )


def get_api_key():
    try:
        api_key = st.secrets.get("OLLAMA_API_KEY")
    except Exception:
        api_key = None

    if api_key:
        return api_key

    env_key = os.getenv("OLLAMA_API_KEY")

    if env_key:
        return env_key

    return st.sidebar.text_input(
        "OLLAMA_API_KEY",
        type="password",
    )


def get_claim_rows(
    claim_ids,
    claims,
):
    claim_by_id = {claim["claim_id"]: claim for claim in claims}

    rows = []

    for claim_id in claim_ids:
        if claim_id in claim_by_id:
            claim = claim_by_id[claim_id]

            rows.append(
                {
                    "claim_id": claim_id,
                    "text": claim["text"],
                    "document_id": claim["document_id"],
                }
            )

    return rows


def run_model(
    client,
    mode,
    question,
    documents,
    claims,
    stress_test="Нет",
    stress_claim_id=None,
    expected_status=None,
):
    context, top_claims, allowed_ids = build_context(
        mode,
        question,
        documents,
        claims,
        stress_claim_id,
    )

    start_time = time.time()

    answer, token_count = ask_llm(
        client=client,
        question_id=question["question_id"],
        question=question["question"],
        context=context,
    )

    response_time = time.time() - start_time

    errors = validate_answer(
        answer,
        allowed_ids,
    )

    answer_dict = answer.model_dump()

    metrics = evaluate_answer(
        question=question,
        answer=answer_dict,
        token_count=token_count,
        response_time=response_time,
    )

    if expected_status is None:
        expected_status = question["gold_status"]

    result = {
        "question_id": question["question_id"],
        "mode": mode,
        "stress_test": stress_test,
        "expected_status": expected_status,
        "gold_status": question["gold_status"],
        "status": answer.status,
        "answer": answer.answer,
        "evidence_ids": answer.evidence_ids,
        "counterevidence_ids": answer.counterevidence_ids,
        "confidence": answer.confidence,
        "status_accuracy": metrics["status_accuracy"],
        "status_correct": int(answer.status == expected_status),
        "evidence_precision": metrics["evidence_precision"],
        "evidence_recall": metrics["evidence_recall"],
        "evidence_f1": metrics["evidence_f1"],
        "counterevidence_recall": metrics["counterevidence_recall"],
        "confident_answer_when_insufficient": metrics[
            "confident_answer_when_insufficient"
        ],
        "token_count": token_count,
        "response_time": response_time,
    }

    return (
        answer,
        result,
        errors,
        context,
        top_claims,
    )


def run_robustness(
    client,
    question,
    documents,
    claims,
):
    all_results = []
    robustness_rows = []
    base_results = {}

    for mode in ["A", "B", "C"]:
        answer, base_result, errors, context, top_claims = run_model(
            client=client,
            mode=mode,
            question=question,
            documents=documents,
            claims=claims,
            stress_test="Нет",
            expected_status=question["gold_status"],
        )

        all_results.append(base_result)
        base_results[mode] = base_result

        shuffled_documents = copy_rows(documents)

        random_generator = random.Random(42)

        random_generator.shuffle(shuffled_documents)

        (
            answer,
            reorder_result,
            errors,
            context,
            top_claims,
        ) = run_model(
            client=client,
            mode=mode,
            question=question,
            documents=shuffled_documents,
            claims=claims,
            stress_test="Перестановка документов",
            expected_status=question["gold_status"],
        )

        all_results.append(reorder_result)

        reorder_metrics = evaluate_stress_test(
            base_result,
            reorder_result,
        )

        robustness_rows.append(
            {
                "mode": mode,
                "stress_test": "Перестановка документов",
                "status_changed": reorder_metrics["status_changed"],
                "answer_changed": reorder_metrics["answer_changed"],
                "evidence_stability": reorder_metrics["evidence_stability"],
                "reaction_correct": int(reorder_metrics["answer_changed"] == 0),
            }
        )

        if question["question_id"] in DELETION_QUESTIONS:
            removed_id = question["gold_evidence_ids"][0]

            (
                removed_documents,
                removed_claims,
            ) = remove_claim(
                removed_id,
                documents,
                claims,
            )

            (
                answer,
                deletion_result,
                errors,
                context,
                top_claims,
            ) = run_model(
                client=client,
                mode=mode,
                question=question,
                documents=removed_documents,
                claims=removed_claims,
                stress_test="Удаление посылки",
                expected_status="insufficient",
            )

            all_results.append(deletion_result)

            deletion_metrics = evaluate_stress_test(
                base_result,
                deletion_result,
            )

            robustness_rows.append(
                {
                    "mode": mode,
                    "stress_test": "Удаление посылки",
                    "status_changed": deletion_metrics["status_changed"],
                    "answer_changed": deletion_metrics["answer_changed"],
                    "evidence_stability": deletion_metrics["evidence_stability"],
                    "reaction_correct": int(
                        deletion_result["status"]
                        in {
                            "insufficient",
                            "both",
                        }
                    ),
                }
            )

        (
            contradiction_documents,
            contradiction_claims,
            stress_claim_id,
        ) = add_contradiction(
            question["question_id"],
            documents,
            claims,
        )

        (
            answer,
            contradiction_result,
            errors,
            context,
            top_claims,
        ) = run_model(
            client=client,
            mode=mode,
            question=question,
            documents=contradiction_documents,
            claims=contradiction_claims,
            stress_test="Добавление противоречия",
            stress_claim_id=stress_claim_id,
            expected_status="both",
        )

        all_results.append(contradiction_result)

        contradiction_metrics = evaluate_stress_test(
            base_result,
            contradiction_result,
        )

        robustness_rows.append(
            {
                "mode": mode,
                "stress_test": "Добавление противоречия",
                "status_changed": contradiction_metrics["status_changed"],
                "answer_changed": contradiction_metrics["answer_changed"],
                "evidence_stability": contradiction_metrics["evidence_stability"],
                "reaction_correct": int(contradiction_result["status"] == "both"),
            }
        )

    flat_padic = compare_flat_padic(
        base_results["B"],
        base_results["C"],
    )

    return (
        all_results,
        robustness_rows,
        flat_padic,
    )


if "saved_results" not in st.session_state:
    st.session_state.saved_results = []

if "robustness_results" not in st.session_state:
    st.session_state.robustness_results = []

if "flat_padic_result" not in st.session_state:
    st.session_state.flat_padic_result = None


st.sidebar.header("Досье")

use_uploaded = st.sidebar.checkbox("Загрузить свои JSONL-файлы")


if use_uploaded:
    uploaded_documents = st.sidebar.file_uploader(
        "documents.jsonl",
        type=["jsonl"],
    )

    uploaded_claims = st.sidebar.file_uploader(
        "claims.jsonl",
        type=["jsonl"],
    )

    uploaded_questions = st.sidebar.file_uploader(
        "questions.jsonl",
        type=["jsonl"],
    )

    if uploaded_documents and uploaded_claims and uploaded_questions:
        documents = load_uploaded_jsonl(uploaded_documents)

        claims = load_uploaded_jsonl(uploaded_claims)

        questions = load_uploaded_jsonl(uploaded_questions)

    else:
        st.info("Загрузите все три файла.")

        st.stop()

else:
    documents = load_jsonl("data/documents.jsonl")

    claims = load_jsonl("data/claims.jsonl")

    questions = load_jsonl("data/questions.jsonl")


question_labels = {
    question["question_id"]: (f"{question['question_id']} — " f"{question['question']}")
    for question in questions
}


left, right = st.columns(2)

with left:
    question_id = st.selectbox(
        "Вопрос",
        list(question_labels),
        format_func=lambda value: question_labels[value],
    )

with right:
    mode = st.selectbox(
        "Режим",
        ["A", "B", "C"],
        format_func=lambda value: {
            "A": "A — полный контекст",
            "B": "B — BM25 top-15",
            "C": "C — p-adic context",
        }[value],
    )


question = next(item for item in questions if item["question_id"] == question_id)


st.subheader("Стресс-тест")

stress_test = st.selectbox(
    "Изменение контекста",
    [
        "Нет",
        "Перестановка документов",
        "Удаление посылки",
        "Добавление противоречия",
    ],
)


working_documents = copy_rows(documents)

working_claims = copy_rows(claims)

stress_claim_id = None

expected_status = question["gold_status"]


if stress_test == "Перестановка документов":
    random_generator = random.Random(42)

    random_generator.shuffle(working_documents)

    st.write(
        "Новый порядок документов:",
        [document["document_id"] for document in working_documents],
    )


elif stress_test == "Удаление посылки":
    if question_id in DELETION_QUESTIONS:
        available_ids = question["gold_evidence_ids"]

        removed_id = st.selectbox(
            "Какую посылку удалить",
            available_ids,
        )

        (
            working_documents,
            working_claims,
        ) = remove_claim(
            removed_id,
            working_documents,
            working_claims,
        )

        expected_status = "insufficient"

        st.write(
            "Удалена посылка:",
            removed_id,
        )

    else:
        st.info("Удаление посылки применяется " "к multi-hop и временным вопросам.")


elif stress_test == "Добавление противоречия":
    (
        working_documents,
        working_claims,
        stress_claim_id,
    ) = add_contradiction(
        question_id,
        working_documents,
        working_claims,
    )

    expected_status = "both"

    st.write(
        "Добавлено противоречие:",
        CONTRADICTIONS[question_id],
    )


st.write(
    "Ожидаемый статус:",
    expected_status,
)


(
    context,
    top_claims,
    allowed_ids,
) = build_context(
    mode,
    question,
    working_documents,
    working_claims,
    stress_claim_id,
)


if mode in ["B", "C"]:
    st.write(
        "Claim ID в retrieval:",
        [claim["claim_id"] for claim in top_claims],
    )


with st.expander("Контекст модели"):
    st.text(context)


with st.expander("p-adic ветви"):
    documents_by_id = {
        document["document_id"]: document for document in working_documents
    }

    padic_preview = build_padic_context(
        top_claims,
        documents_by_id,
    )

    st.text(padic_preview)


api_key = get_api_key()


if st.button(
    "Получить ответ",
    type="primary",
):
    if not api_key:
        st.error("Не найден OLLAMA_API_KEY.")

    else:
        client = create_client(api_key)

        try:
            (
                answer,
                result,
                errors,
                context,
                top_claims,
            ) = run_model(
                client=client,
                mode=mode,
                question=question,
                documents=working_documents,
                claims=working_claims,
                stress_test=stress_test,
                stress_claim_id=stress_claim_id,
                expected_status=expected_status,
            )

            st.subheader("Ответ модели")

            st.json(answer.model_dump())

            (
                col_1,
                col_2,
                col_3,
                col_4,
            ) = st.columns(4)

            col_1.metric(
                "Status",
                answer.status,
            )

            col_2.metric(
                "Confidence",
                round(
                    answer.confidence,
                    3,
                ),
            )

            col_3.metric(
                "Токены",
                result["token_count"],
            )

            col_4.metric(
                "Время",
                f"{result['response_time']:.2f} с",
            )

            evidence_rows = get_claim_rows(
                answer.evidence_ids,
                working_claims,
            )

            counterevidence_rows = get_claim_rows(
                answer.counterevidence_ids,
                working_claims,
            )

            st.subheader("Доказательства")

            if evidence_rows:
                st.dataframe(
                    pd.DataFrame(evidence_rows),
                    width="stretch",
                )
            else:
                st.write("Доказательства отсутствуют.")

            st.subheader("Контрдоказательства")

            if counterevidence_rows:
                st.dataframe(
                    pd.DataFrame(counterevidence_rows),
                    width="stretch",
                )
            else:
                st.write("Контрдоказательства отсутствуют.")

            metrics_table = {
                "Status accuracy": result["status_accuracy"],
                "Evidence precision": result["evidence_precision"],
                "Evidence recall": result["evidence_recall"],
                "Evidence F1": result["evidence_f1"],
                "Counterevidence recall": result["counterevidence_recall"],
            }

            st.subheader("Метрики")

            st.dataframe(
                pd.DataFrame([metrics_table]),
                width="stretch",
            )

            if errors:
                st.warning(errors)
            else:
                st.success("Формат ответа прошёл проверку.")

            st.session_state.saved_results.append(result)

        except Exception as error:
            st.error(f"Ошибка LLM: {error}")


st.subheader("Автоматическая проверка устойчивости")

st.write("Проверяются режимы A, B и C.")

if st.button("Запустить тест устойчивости A/B/C"):
    if not api_key:
        st.error("Не найден OLLAMA_API_KEY.")

    else:
        client = create_client(api_key)

        try:
            with st.spinner("Выполняются тесты..."):
                (
                    robustness_runs,
                    robustness_rows,
                    flat_padic_result,
                ) = run_robustness(
                    client=client,
                    question=question,
                    documents=documents,
                    claims=claims,
                )

            st.session_state.saved_results.extend(robustness_runs)

            st.session_state.robustness_results = robustness_rows

            st.session_state.flat_padic_result = flat_padic_result

            st.success("Тест устойчивости завершён.")

        except Exception as error:
            st.error(f"Ошибка теста устойчивости: {error}")


if st.session_state.robustness_results:
    robustness_df = pd.DataFrame(st.session_state.robustness_results)

    st.dataframe(
        robustness_df,
        width="stretch",
    )

    reorder_rows = robustness_df[
        robustness_df["stress_test"] == "Перестановка документов"
    ]

    deletion_rows = robustness_df[robustness_df["stress_test"] == "Удаление посылки"]

    contradiction_rows = robustness_df[
        robustness_df["stress_test"] == "Добавление противоречия"
    ]

    summary = {
        "Доля изменившихся после перестановки": (
            reorder_rows["answer_changed"].mean() if len(reorder_rows) > 0 else 0.0
        ),
        "Правильная реакция на удаление": (
            deletion_rows["reaction_correct"].mean() if len(deletion_rows) > 0 else None
        ),
        "Правильная реакция на противоречие": (
            contradiction_rows["reaction_correct"].mean()
            if len(contradiction_rows) > 0
            else 0.0
        ),
        "Средняя стабильность evidence_ids": robustness_df["evidence_stability"].mean(),
    }

    st.subheader("Итог устойчивости")

    st.dataframe(
        pd.DataFrame([summary]),
        width="stretch",
    )


if st.session_state.flat_padic_result is not None:
    st.subheader("Разница между B и C")

    st.dataframe(
        pd.DataFrame([st.session_state["flat_padic_result"]]),
        width="stretch",
    )


st.subheader("Сравнение сохранённых результатов")


if st.session_state.saved_results:
    results_df = pd.DataFrame(st.session_state.saved_results)

    question_results_df = results_df[results_df["question_id"] == question_id]

    st.write(f"Результаты для {question_id}")

    st.dataframe(
        question_results_df,
        width="stretch",
    )

    with st.expander("Все сохранённые результаты"):
        st.dataframe(
            results_df,
            width="stretch",
        )

    csv_data = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Скачать результаты CSV",
        data=csv_data,
        file_name="results.csv",
        mime="text/csv",
    )

    if st.button("Очистить результаты"):
        st.session_state.saved_results = []
        st.session_state.robustness_results = []
        st.session_state.flat_padic_result = None
        st.rerun()

else:
    st.info("Пока нет сохранённых результатов.")
