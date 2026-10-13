from dataclasses import dataclass


# Not named TestQuery: pytest tries to collect any class whose name starts with "Test".
@dataclass
class QueryCase:
    label: str
    query: str
    # Ids of the documents this query should find. They are row positions in the
    # first 100 rows of pqa_artificial, so they change if SUBSET_SIZE or the dataset does.
    targets: list[int]
    # Whether each retriever puts the targets in its top 5. These record what the
    # models actually do today, including their blind spots.
    dense_finds: bool
    sparse_finds: bool


QUERY_CASES = [
    QueryCase("cell-line-name", "MC3T3", [63], False, True),
    QueryCase("gene-variant-id", "rs2301228", [11], False, True),
    QueryCase("marker-with-special-char", "34βE12", [33], False, True),
    QueryCase("drug-name", "thioridazine", [75], True, True),
    QueryCase("synonym", "artificial cornea", [15], True, False),
    QueryCase(
        "paraphrase",
        "Does being served more food make people eat faster?",
        [16],
        True,
        True,
    ),
    QueryCase(
        "layperson-wording",
        "shortness of breath in people hospitalized with a weak heart",
        [44],
        True,
        True,
    ),
    QueryCase("abbreviation", "TBI and drinking", [57], True, True),
    QueryCase("short-keyword", "vitamin D", [30, 55], True, True),
    QueryCase("broad-topic", "breast cancer chemotherapy", [2, 4], True, True),
    QueryCase(
        "long-question",
        "In patients with esophageal cancer, does waiting longer between "
        "chemoradiation and surgery improve the chance of a complete response?",
        [20],
        True,
        True,
    ),
    QueryCase("typo", "hepatocelular carcinoma prognosis", [28, 74], True, True),
]

# Nothing in the corpus is relevant to this one.
OUT_OF_CORPUS_QUERY = "COVID-19 vaccine side effects"
