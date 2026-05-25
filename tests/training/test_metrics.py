import pytest

from sba_uas.training.metrics import (
    average_forgetting,
    average_performance,
    forward_transfer,
)


def test_average_performance_averages_final_model_scores():
    assert average_performance([60.0, 70.0, 80.0]) == pytest.approx(70.0)


def test_average_forgetting_matches_paper_formula():
    score_matrix = [
        [90.0, 12.0, 8.0],
        [82.0, 75.0, 20.0],
        [70.0, 68.0, 65.0],
    ]

    # ((max(90, 82) - 70) + (max(12, 75) - 68)) / 2
    assert average_forgetting(score_matrix) == pytest.approx(13.5)


def test_average_forgetting_can_clamp_negative_transfer_terms():
    score_matrix = [
        [50.0, 10.0],
        [55.0, 60.0],
    ]

    assert average_forgetting(score_matrix) == pytest.approx(-5.0)
    assert average_forgetting(score_matrix, clamp_at_zero=True) == pytest.approx(0.0)


def test_forward_transfer_compares_pretraining_score_with_random_baseline():
    score_matrix = [
        [90.0, 30.0, 12.0],
        [82.0, 75.0, 28.0],
        [70.0, 68.0, 65.0],
    ]
    random_baseline = [0.0, 20.0, 10.0]

    # ((Y_1,2 - b_2) + (Y_2,3 - b_3)) / 2
    assert forward_transfer(score_matrix, random_baseline) == pytest.approx(14.0)


def test_metrics_reject_missing_or_invalid_scores():
    with pytest.raises(ValueError, match="score_matrix must be square"):
        average_forgetting([[1.0, 2.0], [3.0]])

    with pytest.raises(ValueError, match="finite"):
        average_performance([1.0, float("nan")])

    with pytest.raises(ValueError, match="missing task indices"):
        forward_transfer([[1.0, 2.0], [3.0, 4.0]], {0: 0.0})
