from pathlib import Path

p = Path("startscale_shadow_controller.py")
s = p.read_text(encoding="utf-8")

marker = """# ============================================================
# SAVE PREDICTION
# ============================================================"""

block = """# ============================================================
# REAL TRACE ETA EVALUATION (OFFLINE SHADOW ANALYSIS)
# ============================================================

TRACE_FILE = "azure_replay_20min.csv"
TRACE_MIN_RPS = 25.0
TRACE_MAX_RPS = 40.0
ETA_SMOOTHING_WINDOW_SEC = 10
ETA_HORIZON_SEC = 30

trace_df = pd.read_csv(TRACE_FILE)

trace_df["smoothed_rps"] = (
    trace_df["requests_per_second"]
    .rolling(ETA_SMOOTHING_WINDOW_SEC, min_periods=1)
    .mean()
)

original_min_rps = trace_df["requests_per_second"].min()
original_max_rps = trace_df["requests_per_second"].max()

trace_df["normalized_rps"] = (
    TRACE_MIN_RPS
    + (trace_df["requests_per_second"] - original_min_rps)
    / (original_max_rps - original_min_rps)
    * (TRACE_MAX_RPS - TRACE_MIN_RPS)
)

trace_df["smoothed_normalized_rps"] = (
    trace_df["normalized_rps"]
    .rolling(ETA_SMOOTHING_WINDOW_SEC, min_periods=1)
    .mean()
)

required_lead_time_ms = calibrated_p90 + 500

eta_rows = []

for i in range(len(trace_df) - ETA_HORIZON_SEC):

    current_rps = float(
        trace_df.loc[i, "smoothed_normalized_rps"]
    )

    future_window = trace_df.loc[
        i + 1:i + ETA_HORIZON_SEC,
        "smoothed_normalized_rps"
    ]

    crossing = future_window[
        future_window >= TRACE_MAX_RPS
    ]

    if len(crossing) > 0:

        first_crossing_index = crossing.index[0]

        demand_eta_sec = float(
            first_crossing_index - i
        )

        demand_eta_ms = (
            demand_eta_sec * 1000.0
        )

        decision, lead_time = scaling_decision(
            demand_eta_ms=demand_eta_ms,
            predicted_p90_ms=calibrated_p90,
        )

        crossing_found = True

    else:

        demand_eta_sec = float("nan")
        demand_eta_ms = float("nan")
        decision = "NO_THRESHOLD_CROSSING"
        lead_time = required_lead_time_ms
        crossing_found = False

    eta_rows.append({
        "second": int(trace_df.loc[i, "second"]),
        "smoothed_rps": current_rps,
        "demand_eta_sec": demand_eta_sec,
        "demand_eta_ms": demand_eta_ms,
        "predicted_p90_ms": calibrated_p90,
        "required_lead_time_ms": lead_time,
        "decision": decision,
        "threshold_crossing_found": crossing_found,
    })

eta_df = pd.DataFrame(eta_rows)

eta_df.to_csv(
    "startscale_shadow_trace_eta.csv",
    index=False,
)

print()
print("=" * 80)
print("REAL TRACE ETA EVALUATION")
print("=" * 80)
print(f"Trace rows              : {len(trace_df)}")
print(f"Trace duration          : {len(trace_df) / 60:.1f} minutes")
print(f"Replay RPS range        : {TRACE_MIN_RPS:.1f}-{TRACE_MAX_RPS:.1f}")
print(f"Smoothing window        : {ETA_SMOOTHING_WINDOW_SEC} seconds")
print(f"ETA look-ahead          : {ETA_HORIZON_SEC} seconds")
print(f"Predicted calibrated P90: {calibrated_p90:.2f} ms")
print(f"Required lead time      : {required_lead_time_ms:.2f} ms")
print(
    f"Crossing events         : "
    f"{int(eta_df['threshold_crossing_found'].sum())}"
)
print()
print("Saved: startscale_shadow_trace_eta.csv")

"""

assert marker in s, "SAVE PREDICTION marker not found"
assert "REAL TRACE ETA EVALUATION" not in s, "ETA block already exists"

p.write_text(
    s.replace(marker, block + marker, 1),
    encoding="utf-8"
)

print("PATCH SUCCESS")
