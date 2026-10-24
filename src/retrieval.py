import re
from rank_bm25 import BM25Okapi

_token_re = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    return _token_re.findall(text.lower())


def get_top_claims(question, claims, top_k):
    if not claims or top_k <= 0:
        return []

    corpus = [tokenize(claim["text"]) for claim in claims]
    bm25 = BM25Okapi(corpus)
    query = tokenize(question)

    if not query:
        return []

    return bm25.get_top_n(query, claims, n=min(top_k, len(claims)))