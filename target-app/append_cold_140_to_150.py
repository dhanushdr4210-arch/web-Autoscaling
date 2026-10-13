import pandas as pd
import numpy as np

MASTER = "v3_master_140_raw.csv"
COLD = "v3_combined_cpu11_14_mem22_23_cold.csv"
OUTPUT = "v3_master_150_raw.csv"

master = pd.read_csv(MASTER)
cold = pd.read_csv(COLD)

print("=" * 70)
print("STARTSCALE — APPEND COLD CPU11–14 / MEM22–23 CELL")
print("=" * 70)

print(f"Master before: {master.shape}")
print(f"Cold data:     {cold.shape}")

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

assert master.shape[0] == 140, "Master must contain exactly 140 rows"
assert cold.shape[0] == 10, "Cold dataset must contain exactly 10 rows"
assert cold["request_to_ready_ms"].notna().all(), "NULL latency detected"

expected_trials = list(range(1, 11))
assert sorted(cold["trial"].astype(int).tolist()) == expected_trials, \
    "Trials are not exactly 1–10"

# Make sure this is really the cold experiment
assert cold["experiment"].eq(
    "V3_Combined_CPU11_14_Mem22_23_COLD"
).all(), "Unexpected experiment name"

# ------------------------------------------------------------
# Prepare the 10 new rows
# ------------------------------------------------------------

new = pd.DataFrame(index=cold.index)

# Copy fields that exist in the master
for col in master.columns:
    if col in cold.columns:
        new[col] = cold[col]
    else:
        new[col] = np.nan

# Explicit research labels
new["experiment_original"] = (
    "V3_Combined_CPU11_14_Mem22_23_COLD"
)

new["source_condition"] = (
    "COMBINED_CPU11_14_MEM22_23_COLD"
)

# Preserve actual CPU information
new["cpu_num"] = np.nan

# Keep the measured values exactly as recorded
new["pre_scale_node_cpu_pct"] = pd.to_numeric(
    cold["pre_scale_node_cpu_pct"], errors="coerce"
)

new["pre_scale_node_memory_pct"] = pd.to_numeric(
    cold["pre_scale_node_memory_pct"], errors="coerce"
)

new["request_to_ready_ms"] = pd.to_numeric(
    cold["request_to_ready_ms"], errors="coerce"
)

# ------------------------------------------------------------
# Make column order identical to master
# ------------------------------------------------------------

new = new[master.columns]

# ------------------------------------------------------------
# Check for accidental duplicate rows
# ------------------------------------------------------------

combined = pd.concat([master, new], ignore_index=True)

assert combined.shape == (150, 23), \
    f"Unexpected final shape: {combined.shape}"

assert combined["request_to_ready_ms"].notna().all(), \
    "NULL latency after append"

# Exact duplicate row check
duplicates = combined.duplicated().sum()

print(f"\nExact duplicate rows: {duplicates}")

assert duplicates == 0, \
    "Duplicate rows detected"

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

combined.to_csv(OUTPUT, index=False)

# ------------------------------------------------------------
# Report the new cell
# ------------------------------------------------------------

print("\nNEW COLD CELL")
print("-" * 70)

print(
    combined[
        combined["source_condition"]
        == "COMBINED_CPU11_14_MEM22_23_COLD"
    ][
        [
            "trial",
            "pre_scale_node_cpu_pct",
            "pre_scale_node_memory_pct",
            "request_to_ready_ms"
        ]
    ].to_string(index=False)
)

cell = combined[
    combined["source_condition"]
    == "COMBINED_CPU11_14_MEM22_23_COLD"
]

print("\nCOLD CELL STATISTICS")
print("-" * 70)
print(f"n      : {len(cell)}")
print(f"mean   : {cell['request_to_ready_ms'].mean():.2f} ms")
print(f"median : {cell['request_to_ready_ms'].median():.2f} ms")
print(f"std    : {cell['request_to_ready_ms'].std():.2f} ms")
print(f"min    : {cell['request_to_ready_ms'].min():.2f} ms")
print(f"max    : {cell['request_to_ready_ms'].max():.2f} ms")

# Strict nominal-cell membership
strict = cell[
    cell["pre_scale_node_cpu_pct"].between(11, 14, inclusive="both")
    & cell["pre_scale_node_memory_pct"].between(22, 23, inclusive="both")
]

print("\nSTRICT 11–14% CPU / 22–23% MEMORY SUBSET")
print("-" * 70)
print(f"Rows inside strict cell: {len(strict)}/10")

if len(strict):
    print(f"Mean   : {strict['request_to_ready_ms'].mean():.2f} ms")
    print(f"Median : {strict['request_to_ready_ms'].median():.2f} ms")
    print(f"Std    : {strict['request_to_ready_ms'].std():.2f} ms")

print("\nFINAL MASTER")
print("-" * 70)
print(f"Saved: {OUTPUT}")
print(f"Shape: {combined.shape}")

print("\nDONE")