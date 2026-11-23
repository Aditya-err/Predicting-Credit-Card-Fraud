import pytest
import numpy as np
from src.ensembling import to_rank_percentiles, compute_weighted_probability_blend, compute_rank_average_blend, optimize_ensemble_weights

def test_to_rank_percentiles():
    arr = np.array([10.0, 50.0, 20.0, 100.0])
    ranks = to_rank_percentiles(arr)
    assert ranks.min() == 0.0
    assert ranks.max() == 1.0
    assert ranks[0] < ranks[2] < ranks[1] < ranks[3]

def test_rank_average_blend():
    p1 = np.array([0.1, 0.4, 0.8])
    p2 = np.array([0.05, 0.5, 0.9])
    blend = compute_rank_average_blend([p1, p2], weights=[0.5, 0.5])
    assert len(blend) == 3
    assert blend[0] < blend[1] < blend[2]
    assert 0.0 <= blend.min() <= blend.max() <= 1.0

def test_optimize_ensemble_weights():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    m1 = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9, 0.2, 0.6])
    m2 = np.array([0.2, 0.1, 0.8, 0.7, 0.4, 0.85, 0.1, 0.7])
    
    weights, best_auc = optimize_ensemble_weights([m1, m2], y_true, method="rank")
    assert len(weights) == 2
    assert np.isclose(sum(weights), 1.0)
    assert best_auc >= 0.80
