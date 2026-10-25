from src.padic_config import SOURCE_LEVEL, STATUS_LEVEL, TIME_LEVEL


def get_padic_path(claim, documents):
    document = documents[claim["document_id"]]

    a0 = SOURCE_LEVEL[document["source_type"]]
    a1 = STATUS_LEVEL[claim["status"]]
    a2 = TIME_LEVEL.get(claim["claim_id"], 2)

    return [a0, a1, a2]


def get_padic_code(path):
    return path[0] + path[1] * 3 + path[2] * 9


def padic_distance(path_1, path_2):
    if path_1 == path_2:
        return 0.0

    common = 0

    for value_1, value_2 in zip(path_1, path_2):
        if value_1 != value_2:
            break

        common += 1

    return 3 ** (-common)
