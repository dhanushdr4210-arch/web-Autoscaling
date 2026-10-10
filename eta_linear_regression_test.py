
import csv
from pathlib import Path

TRACE_FILE = Path("azure_replay_20min.csv")
BASELINE_A_FILE = Path("baseline_a_results.csv")

MIN_RPS = 30.0
MAX_RPS = 40.0
WINDOW = 10
THRESHOLD = 40.0
HORIZON = 30


def regression_slope(values):
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in enumerate(values)
    )
    denominator = sum(
        (x - x_mean) ** 2 for x in range(n)
    )

    return numerator / denominator if denominator else 0.0


def main():
    for path in (TRACE_FILE, BASELINE_A_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with BASELINE_A_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        baseline_a = list(csv.DictReader(f))

    target_seconds = {int(row["Second"]) for row in baseline_a}

    with TRACE_FILE.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        trace = list(csv.DictReader(f))

    raw = [float(row["requests_per_second"]) for row in trace]
    low, high = min(raw), max(raw)

    if high == low:
        raise ValueError("Cannot normalize a constant trace.")

    # Offline diagnostic normalization; not suitable for online use.
    values = [
        MIN_RPS + (v - low) / (high - low) * (MAX_RPS - MIN_RPS)
        for v in raw
    ]

    baseline_a_by_second = {
        int(row["Second"]): row for row in baseline_a
    }

    comparison = []

    for i in sorted(target_seconds):
        if i < WINDOW - 1 or i + HORIZON >= len(values):
            continue

        past = values[i - WINDOW + 1:i + 1]
        slope = regression_slope(past)

        actual_eta = next(
            (
                step for step in range(1, HORIZON + 1)
                if values[i + step] >= THRESHOLD
            ),
            None,
        )

        if actual_eta is None:
            continue

        if slope <= 0:
            predicted_b = float(HORIZON)
        else:
            predicted_b = max(
                0.0,
                min(float(HORIZON), (THRESHOLD - values[i]) / slope),
            )

        a_row = baseline_a_by_second[i]
        predicted_a = float(a_row["PredictedETA"])
        actual_a = int(float(a_row["ActualETA"]))

        # Verify both methods use the same label.
        if actual_a != actual_eta:
            print(f"Label mismatch at second {i}; skipping.")
            continue

        comparison.append({
            "Second": i,
            "ActualETA": actual_eta,
            "BaselineA_ETA": predicted_a,
            "BaselineA_Error": abs(predicted_a - actual_eta),
            "BaselineB_ETA": predicted_b,
            "BaselineB_Error": abs(predicted_b - actual_eta),
        })

    if not comparison:
        print("No matching evaluation windows found.")
        return

    print("Same-window comparison")
    print(
        f"{'Second':>6} {'Actual':>7} {'A ETA':>8} "
        f"{'A Error':>8} {'B ETA':>8} {'B Error':>8}"
    )

    for row in comparison:
        print(
            f"{row['Second']:>6} {row['ActualETA']:>7} "
            f"{row['BaselineA_ETA']:>8.2f} "
            f"{row['BaselineA_Error']:>8.2f} "
            f"{row['BaselineB_ETA']:>8.2f} "
            f"{row['BaselineB_Error']:>8.2f}"
        )

    mae_a = sum(r["BaselineA_Error"] for r in comparison) / len(comparison)
    mae_b = sum(r["BaselineB_Error"] for r in comparison) / len(comparison)

    print(f"\nMatched windows: {len(comparison)}")
    print(f"Baseline A MAE: {mae_a:.2f} seconds")
    print(f"Baseline B MAE: {mae_b:.2f} seconds")
    print(f"MAE difference (A - B): {mae_a - mae_b:.2f} seconds")


if __name__ == "__main__":
    main()