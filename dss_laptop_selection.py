"""Multi-criteria Decision Support System.

Input -> Validation -> Normalization -> WSM -> TOPSIS -> Ranking -> Visualization
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


ALTERNATIVES = [
    "A1 - MacBook Air M4",
    "A2 - ThinkPad X1 Carbon",
    "A3 - ASUS Zenbook 14 OLED",
    "A4 - Dell XPS 14",
    "A5 - Lenovo Legion Slim 5",
    "A6 - HP Spectre x360 14",
]

CRITERIA = ["price", "performance", "battery", "weight", "ram", "dev_experience"]

CRITERIA_TYPES = {
    "price": "cost",
    "performance": "benefit",
    "battery": "benefit",
    "weight": "cost",
    "ram": "benefit",
    "dev_experience": "benefit",
}

WEIGHTS = {
    "price": 0.25,
    "performance": 0.25,
    "battery": 0.15,
    "weight": 0.10,
    "ram": 0.10,
    "dev_experience": 0.15,
}

DECISION_MATRIX = pd.DataFrame(
    [
        [950, 8.5, 15, 1.24, 16, 8.5],
        [1050, 7.5, 11, 1.12, 16, 9.0],
        [700, 7.0, 12, 1.28, 16, 8.0],
        [1150, 8.0, 10, 1.68, 16, 8.5],
        [1100, 9.5, 7, 2.10, 32, 8.0],
        [1000, 7.5, 12, 1.44, 16, 8.5],
    ],
    index=ALTERNATIVES,
    columns=CRITERIA,
)


def validate_inputs(
    matrix: pd.DataFrame,
    weights: dict[str, float],
    criteria_types: dict[str, str],
) -> None:
    """Validate the decision matrix, weights and criterion types."""
    if matrix.isna().any().any():
        raise ValueError("Decision matrix contains missing values.")

    if not all(pd.api.types.is_numeric_dtype(matrix[col]) for col in matrix.columns):
        raise ValueError("All criteria columns must be numeric.")

    if set(matrix.columns) != set(weights):
        raise ValueError("Weights must contain exactly the same criteria as the matrix.")

    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"Sum of weights must be 1.0, got {sum(weights.values()):.4f}.")

    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Weights must be non-negative.")

    if set(matrix.columns) != set(criteria_types):
        raise ValueError("Criteria types must contain exactly the same criteria as the matrix.")

    invalid_types = set(criteria_types.values()) - {"benefit", "cost"}
    if invalid_types:
        raise ValueError(f"Invalid criterion types: {invalid_types}.")


def normalize(matrix: pd.DataFrame, criteria_types: dict[str, str]) -> pd.DataFrame:
    """Normalize the decision matrix.

    For a benefit criterion:
        x_norm = x / max(x)

    For a cost criterion:
        x_norm = min(x) / x
    """
    normalized = pd.DataFrame(index=matrix.index, columns=matrix.columns, dtype=float)

    for criterion in matrix.columns:
        values = matrix[criterion].astype(float)

        if criteria_types[criterion] == "benefit":
            normalized[criterion] = values / values.max()
        else:
            normalized[criterion] = values.min() / values

    return normalized


def weighted_sum_model(normalized: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Calculate Weighted Sum Model scores."""
    weight_vector = pd.Series(weights)
    return normalized.mul(weight_vector, axis=1).sum(axis=1)


def topsis(
    matrix: pd.DataFrame,
    weights: dict[str, float],
    criteria_types: dict[str, str],
) -> pd.Series:
    """Calculate TOPSIS closeness coefficients."""
    normalized = matrix.astype(float) / np.sqrt((matrix.astype(float) ** 2).sum(axis=0))
    weighted = normalized.mul(pd.Series(weights), axis=1)

    ideal_best = {}
    ideal_worst = {}

    for criterion in weighted.columns:
        if criteria_types[criterion] == "benefit":
            ideal_best[criterion] = weighted[criterion].max()
            ideal_worst[criterion] = weighted[criterion].min()
        else:
            ideal_best[criterion] = weighted[criterion].min()
            ideal_worst[criterion] = weighted[criterion].max()

    ideal_best = pd.Series(ideal_best)
    ideal_worst = pd.Series(ideal_worst)

    distance_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    return distance_worst / (distance_best + distance_worst)


def rank_series(values: pd.Series, ascending: bool = False) -> pd.Series:
    """Return 1-based ranks; rank 1 is the best alternative."""
    return values.rank(ascending=ascending, method="min").astype(int)


def plot_scores(scores: pd.Series, title: str, filename: str) -> None:
    ax = scores.sort_values().plot(kind="barh", color="#4C78A8", figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Alternative")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_ranking(results: pd.DataFrame, filename: str) -> None:
    ax = results[["WSM Rank", "TOPSIS Rank"]].plot(
        kind="bar",
        figsize=(10, 5),
        color=["#4C78A8", "#F58518"],
    )
    ax.set_title("WSM vs TOPSIS ranking")
    ax.set_xlabel("Alternative")
    ax.set_ylabel("Rank (1 = best)")
    ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main() -> None:
    os.makedirs("output", exist_ok=True)
    validate_inputs(DECISION_MATRIX, WEIGHTS, CRITERIA_TYPES)

    normalized = normalize(DECISION_MATRIX, CRITERIA_TYPES)
    wsm_scores = weighted_sum_model(normalized, WEIGHTS)
    topsis_scores = topsis(DECISION_MATRIX, WEIGHTS, CRITERIA_TYPES)

    results = pd.DataFrame(
        {
            "WSM Score": wsm_scores,
            "WSM Rank": rank_series(wsm_scores),
            "TOPSIS Score": topsis_scores,
            "TOPSIS Rank": rank_series(topsis_scores),
        }
    ).sort_values("WSM Rank")

    print("Decision matrix:")
    print(DECISION_MATRIX, end="\n\n")

    print("Normalized decision matrix:")
    print(normalized.round(4), end="\n\n")

    print("Results and ranking:")
    print(results.round(4), end="\n\n")

    wsm_winner = results["WSM Rank"].idxmin()
    topsis_winner = results["TOPSIS Rank"].idxmin()

    print(f"WSM winner: {wsm_winner}")
    print(f"TOPSIS winner: {topsis_winner}")

    if wsm_winner == topsis_winner:
        print("Both methods select the same winner.")
    else:
        print("The methods select different winners.")

    if results["WSM Rank"].equals(results["TOPSIS Rank"]):
        print("The full rankings are identical.")
    else:
        print(
            "The rankings differ because WSM uses a linear weighted sum, "
            "while TOPSIS measures the distance to ideal and anti-ideal alternatives."
        )

    normalized.round(4).to_csv("output/normalized_matrix.csv")
    results.round(4).to_csv("output/results.csv")

    plot_scores(
        wsm_scores,
        "WSM scores",
        "output/wsm_scores.png",
    )
    plot_scores(
        topsis_scores,
        "TOPSIS closeness coefficients",
        "output/topsis_scores.png",
    )
    plot_ranking(results, "output/ranking_comparison.png")


if __name__ == "__main__":
    main()
