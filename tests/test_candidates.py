import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from qhyper.candidate_proposal import (candidate_coverage, is_clique,
                                       propose_candidates, recall_at_budget)
from qhyper.projection import project_hypergraph
from qhyper.synthetic import (SyntheticConfig, generate_synthetic,
                              triangle_ambiguity_example)


def test_budgeted_candidates_are_reproducible_unique_and_cliques():
    sample = generate_synthetic(SyntheticConfig(case="S1", n_nodes=60,
                                                num_hyperedges=40), seed=8)
    A = sample.graph.observed().adjacency_csr
    first = propose_candidates(A, budget=120, min_k=2, max_k=4, seed=4)
    second = propose_candidates(A, budget=120, min_k=2, max_k=4, seed=4)
    assert first == second
    assert len(first) <= 120
    assert len(first) == len(set(first))
    assert all(is_clique(A, edge) for edge in first)
    assert set(map(len, first)).issubset({2, 3, 4})


def test_candidate_coverage_is_oracle_recall_upper_bound():
    first, _ = triangle_ambiguity_example()
    A = first.observed().adjacency_csr
    candidates = propose_candidates(A, budget=4, min_k=2, max_k=3, seed=2)
    report = candidate_coverage(candidates, first.hyperedges_truth)
    truth = set(first.hyperedges_truth)
    selected = [edge for edge in candidates if edge in truth]
    actual_recall = len(selected) / len(truth)
    assert actual_recall <= report["recall"]
    curve = recall_at_budget(candidates, first.hyperedges_truth, [0, 1, 4])
    assert curve[0]["recall"] == 0
    assert curve[-1]["recall"] == report["recall"]


def test_size_two_coverage_missing_when_min_size_is_three():
    truth = [(0, 1), (0, 1, 2)]
    A = project_hypergraph(truth, 3)
    candidates = propose_candidates(A, budget=10, min_k=3, max_k=3, seed=1)
    coverage = candidate_coverage(candidates, truth)
    assert coverage["by_size"][2]["recall"] == 0
    assert coverage["by_size"][3]["recall"] == 1


def test_explicit_per_size_budgets_are_hard_caps():
    A = project_hypergraph([(0, 1, 2, 3, 4)], 5)
    candidates = propose_candidates(A, budget=10, min_k=2, max_k=4, seed=1,
                                    per_size_budget={2: 2, 3: 3, 4: 1})
    assert sum(len(e) == 2 for e in candidates) <= 2
    assert sum(len(e) == 3 for e in candidates) <= 3
    assert sum(len(e) == 4 for e in candidates) <= 1


def test_truth_changes_do_not_change_proposals_or_feature_payload():
    first, second = triangle_ambiguity_example()
    one = propose_candidates(first.observed().adjacency_csr, budget=4,
                             min_k=2, max_k=3, seed=3, graph_id="same",
                             return_records=True)
    two = propose_candidates(second.observed().adjacency_csr, budget=4,
                             min_k=2, max_k=3, seed=3, graph_id="same",
                             return_records=True)
    assert [x.feature_payload() for x in one] == [x.feature_payload() for x in two]


def test_candidate_sources_stable_across_python_hash_seeds():
    code = (
        "import json\n"
        "from qhyper.projection import project_hypergraph\n"
        "from qhyper.candidate_proposal import propose_candidates\n"
        "A = project_hypergraph([(0,1,2,3,4)], 5)\n"
        "records = propose_candidates(A, budget=20, min_k=2, max_k=4, "
        "seed=3, return_records=True)\n"
        "print(json.dumps([r.feature_payload() for r in records], sort_keys=True))\n"
    )
    outputs = []
    for hash_seed in ("1", "73"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        outputs.append(subprocess.check_output([sys.executable, "-c", code],
                                               env=env, text=True))
    assert json.loads(outputs[0]) == json.loads(outputs[1])


def test_generator_channels_are_explicit_and_unknown_names_fail():
    A = project_hypergraph([(0, 1, 2, 3, 4)], 5)
    local = propose_candidates(A, budget=20, min_k=2, max_k=4,
                               generators=["local_clique"], return_records=True)
    maximal = propose_candidates(A, budget=20, min_k=2, max_k=4,
                                 generators=["maximal_clique_subsets"],
                                 return_records=True)
    assert local and maximal
    assert all(record.proposal_sources == ("local_clique",) for record in local)
    assert all(record.proposal_sources == ("maximal_clique",) for record in maximal)
    with pytest.raises(ValueError):
        propose_candidates(A, generators=["random_control"])
    with pytest.raises(ValueError):
        propose_candidates(A, generators=[])
    with pytest.raises(ValueError):
        propose_candidates(A, generators=["local_clique", "local_clique"])
