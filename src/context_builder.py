from collections import defaultdict
from src.padic import TIME_LEVEL, get_padic_path


def build_full_context(documents, claims):
    parts = []

    for document in documents:
        parts.append(
            f"{document['document_id']} — "
            f"{document['title']}\n"
            f"{document['text']}"
        )

    parts.append("\nСПРАВОЧНИК CLAIM_ID")

    for claim in claims:
        parts.append(f"{claim['claim_id']}: {claim['text']}")

    return "\n\n".join(parts)


def build_flat_context(claims):
    return "\n".join(f"{claim['claim_id']}: {claim['text']}" for claim in claims)


def build_padic_context(claims, documents):
    branches = defaultdict(list)

    for claim in claims:
        path = tuple(get_padic_path(claim, documents))
        branches[path].append(claim)

    parts = []

    for path in sorted(branches):
        parts.append(f"ВЕТВЬ {list(path)}")

        branch_claims = sorted(
            branches[path],
            key=lambda claim: (TIME_LEVEL.get(claim["claim_id"], 2), claim["claim_id"]),
        )

        for claim in branch_claims:
            parts.append(f"{claim['claim_id']}: " f"{claim['text']}")

        parts.append("")

    return "\n".join(parts).strip()
