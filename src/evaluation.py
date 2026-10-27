import pandas as pd

CONFIDENCE_THRESHOLD = 0.8


def status_accuracy(gold_status, predicted_status):
    return int(gold_status == predicted_status)


def precision_recall_f1(gold_ids, predicted_ids):
    gold = set(gold_ids)
    predicted = set(predicted_ids)

    if not predicted:
        precision = 0.0
    else:
        precision = len(gold & predicted) / len(predicted)

    if not gold:
        recall = 1.0
    else:
        recall = len(gold & predicted) / len(gold)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def counterevidence_recall(gold_ids, predicted_ids):
    gold = set(gold_ids)
    predicted = set(predicted_ids)

    if not gold:
        return 1.0

    return len(gold & predicted) / len(gold)


def evaluate_answer(question, answer, token_count=0, response_time=0.0):
    precision, recall, f1 = precision_recall_f1(
        question["gold_evidence_ids"], answer["evidence_ids"]
    )

    counter_recall = counterevidence_recall(
        question["gold_counterevidence_ids"], answer["counterevidence_ids"]
    )

    confident_wrong_answer = int(
        question["gold_status"] == "insufficient"
        and answer["status"] != "insufficient"
        and answer["confidence"] >= CONFIDENCE_THRESHOLD
    )

    return {
        "question_id": question["question_id"],
        "gold_status": question["gold_status"],
        "predicted_status": answer["status"],
        "status_accuracy": status_accuracy(question["gold_status"], answer["status"]),
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "counterevidence_recall": counter_recall,
        "confident_answer_when_insufficient": confident_wrong_answer,
        "confidence": answer["confidence"],
        "token_count": token_count,
        "response_time": response_time,
    }


def summarize_results(results):
    if not results:
        return {}

    df = pd.DataFrame(results)

    summary = {
        "status_accuracy": df["status_accuracy"].mean(),
        "evidence_precision": df["evidence_precision"].mean(),
        "evidence_recall": df["evidence_recall"].mean(),
        "evidence_f1": df["evidence_f1"].mean(),
        "counterevidence_recall": df["counterevidence_recall"].mean(),
        "average_token_count": df["token_count"].mean(),
        "average_response_time": df["response_time"].mean(),
    }

    both_rows = df[df["gold_status"] == "both"]

    if len(both_rows) > 0:
        summary["both_accuracy"] = both_rows["status_accuracy"].mean()
    else:
        summary["both_accuracy"] = 0.0

    insufficient_rows = df[df["gold_status"] == "insufficient"]

    if len(insufficient_rows) > 0:
        summary["insufficient_accuracy"] = insufficient_rows["status_accuracy"].mean()

        summary["confident_answer_rate_when_insufficient"] = insufficient_rows[
            "confident_answer_when_insufficient"
        ].mean()
    else:
        summary["insufficient_accuracy"] = 0.0

        summary["confident_answer_rate_when_insufficient"] = 0.0

    return summary


def answer_changed(result_before, result_after):
    return int(
        result_before["status"] != result_after["status"]
        or result_before["answer"] != result_after["answer"]
    )


def evidence_stability(result_before, result_after):
    before = set(result_before["evidence_ids"])

    after = set(result_after["evidence_ids"])

    union = before | after

    if not union:
        return 1.0

    return len(before & after) / len(union)


def evaluate_stress_test(base_result, stress_result):
    return {
        "answer_changed": answer_changed(base_result, stress_result),
        "status_changed": int(base_result["status"] != stress_result["status"]),
        "evidence_stability": evidence_stability(base_result, stress_result),
    }


def compare_flat_padic(flat_result, padic_result):
    return {
        "status_changed": int(flat_result["status"] != padic_result["status"]),
        "answer_changed": int(flat_result["answer"] != padic_result["answer"]),
        "evidence_stability": evidence_stability(flat_result, padic_result),
        "confidence_difference": abs(
            flat_result["confidence"] - padic_result["confidence"]
        ),
    }


def summarize_robustness(results):
    if not results:
        return {}

    total = len(results)

    changed = sum(item["answer_changed"] for item in results)

    average_stability = sum(item["evidence_stability"] for item in results) / total

    return {
        "changed_answer_rate": changed / total,
        "average_evidence_stability": average_stability,
    }
