"""Continual-learning metrics used by the SBA-UAS experiments."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Union


ScoreMatrix = Sequence[Sequence[float]]


def average_performance(final_scores: Sequence[float]) -> float:
    """Compute AP from final-model scores over all evaluation tasks."""

    scores = _finite_sequence(final_scores, name="final_scores")
    return sum(scores) / len(scores)


def average_forgetting(
    score_matrix: ScoreMatrix,
    clamp_at_zero: bool = False,
) -> float:
    """Compute AF from a task-by-task evaluation matrix.

    ``score_matrix[k][i]`` is the score on task ``i`` after completing training
    stage ``k``. The final row is treated as ``Y_T`` in the paper.
    """

    matrix = _finite_square_matrix(score_matrix)
    task_count = len(matrix)
    if task_count < 2:
        raise ValueError("average_forgetting requires at least two tasks")

    final_scores = matrix[-1]
    forgetting_terms = []
    for task_index in range(task_count - 1):
        best_previous = max(row[task_index] for row in matrix[:-1])
        forgetting = best_previous - final_scores[task_index]
        if clamp_at_zero:
            forgetting = max(0.0, forgetting)
        forgetting_terms.append(forgetting)

    return sum(forgetting_terms) / len(forgetting_terms)


def forward_transfer(
    score_matrix: ScoreMatrix,
    random_baseline: Union[Sequence[float], Mapping[int, float]],
) -> float:
    """Compute FWT using scores before each new task is trained.

    The matrix must include evaluation on future tasks. For task ``i > 0``, the
    score after stage ``i - 1`` is compared with the random baseline for task
    ``i``.
    """

    matrix = _finite_square_matrix(score_matrix)
    task_count = len(matrix)
    if task_count < 2:
        raise ValueError("forward_transfer requires at least two tasks")

    baseline = _baseline_lookup(random_baseline, task_count)
    transfer_terms = [
        matrix[task_index - 1][task_index] - baseline[task_index]
        for task_index in range(1, task_count)
    ]
    return sum(transfer_terms) / len(transfer_terms)


def _finite_square_matrix(score_matrix: ScoreMatrix) -> List[List[float]]:
    matrix = [_finite_sequence(row, name="score_matrix row") for row in score_matrix]
    if not matrix:
        raise ValueError("score_matrix cannot be empty")
    width = len(matrix)
    for row in matrix:
        if len(row) != width:
            raise ValueError("score_matrix must be square")
    return matrix


def _finite_sequence(values: Sequence[float], name: str) -> List[float]:
    if not values:
        raise ValueError("{} cannot be empty".format(name))
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("{} must contain only finite numbers".format(name))
    return result


def _baseline_lookup(
    random_baseline: Union[Sequence[float], Mapping[int, float]],
    task_count: int,
) -> Dict[int, float]:
    if isinstance(random_baseline, Mapping):
        baseline = {int(key): float(value) for key, value in random_baseline.items()}
    else:
        values = _finite_sequence(random_baseline, name="random_baseline")
        baseline = {index: value for index, value in enumerate(values)}

    missing = [index for index in range(1, task_count) if index not in baseline]
    if missing:
        raise ValueError(
            "random_baseline is missing task indices {}".format(missing)
        )
    if any(not math.isfinite(baseline[index]) for index in range(1, task_count)):
        raise ValueError("random_baseline must contain only finite numbers")
    return baseline
