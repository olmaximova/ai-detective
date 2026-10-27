from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src.padic import get_padic_code, padic_distance
from src.retrieval import get_top_claims
from src.schemas import ModelAnswer
from src.llm import ask_llm, validate_answer
from src.context_builder import build_flat_context, build_padic_context


def answer(status='entailed', evidence=None, counter=None, text='Ответ'):
    return ModelAnswer(
        question_id='Q01', status=status, answer=text,
        evidence_ids=evidence or [], counterevidence_ids=counter or [],
        confidence=0.9, explanation='Проверяемое объяснение',
    )


def test_same_padic_paths_have_zero_distance():
    assert padic_distance([0, 1, 2], [0, 1, 2]) == 0


@pytest.mark.parametrize('left,right,expected', [
    ([0, 0, 0], [1, 0, 0], 1),
    ([0, 0, 0], [0, 1, 0], 1 / 3),
    ([0, 0, 0], [0, 0, 1], 1 / 9),
])
def test_padic_difference_at_each_level(left, right, expected):
    assert padic_distance(left, right) == pytest.approx(expected)


def test_ultrametric_inequality():
    from itertools import product
    paths = list(product(range(3), repeat=3))
    for x in paths:
        for y in paths:
            for z in paths:
                assert padic_distance(x, z) <= max(
                    padic_distance(x, y), padic_distance(y, z)
                ) + 1e-12


def test_padic_code_is_not_euclidean_distance():
    assert get_padic_code([1, 0, 0]) == 1
    assert get_padic_code([0, 1, 0]) == 3
    assert padic_distance([1, 0, 0], [0, 1, 0]) == 1


def test_invalid_llm_json_is_rejected():
    with pytest.raises(ValidationError):
        ModelAnswer.model_validate_json('{"status": "entailed", invalid}')


def test_unknown_evidence_id_is_rejected():
    errors = validate_answer(answer(evidence=['C999']), {'C01', 'C02'})
    assert any('C999' in error for error in errors)


def test_both_without_counterevidence_is_rejected():
    errors = validate_answer(answer(status='both', evidence=['C01']), {'C01'})
    assert any('counterevidence_ids' in error for error in errors)


def test_insufficient_with_fabricated_answer_is_rejected():
    errors = validate_answer(answer(status='insufficient', text='Виновен участник'), set())
    assert any('null' in error for error in errors)


def test_unavailable_llm_propagates_error(monkeypatch):
    monkeypatch.setattr('src.llm.load_system_prompt', lambda: 'Системный промпт')
    client = Mock()
    client.chat.side_effect = ConnectionError('LLM недоступна')
    with pytest.raises(ConnectionError, match='LLM недоступна'):
        ask_llm(client, 'Q01', 'Что произошло?', 'C01: факт')


def test_bm25_returns_at_most_15_claims():
    claims = [
        {'claim_id': f'C{i:03}', 'text': f'Архив экспортирован участником номер {i}'}
        for i in range(30)
    ]
    selected = get_top_claims('Кто экспортировал архив?', claims, top_k=15)
    assert len(selected) <= 15
    assert len({c['claim_id'] for c in selected}) == len(selected)


def test_flat_and_padic_use_same_claim_ids():
    claims = [
        {'claim_id': 'C01', 'text': 'Архив создан', 'document_id': 'D01', 'status': 'direct'},
        {'claim_id': 'C02', 'text': 'Свидетель заметил архив', 'document_id': 'D02', 'status': 'reported'},
    ]
    documents = {
        'D01': {'document_id': 'D01', 'source_type': 'system_log'},
        'D02': {'document_id': 'D02', 'source_type': 'statement'},
    }
    flat = build_flat_context(claims)
    padic = build_padic_context(claims, documents)
    for claim in claims:
        claim_id = claim['claim_id']
        assert flat.count(f'{claim_id}:') == 1
        assert padic.count(f'{claim_id}:') == 1
    assert {c['claim_id'] for c in claims} == {'C01', 'C02'}
