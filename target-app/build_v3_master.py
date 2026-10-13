import pandas as pd
from pathlib import Path

DATASETS = {
    "CPU_SWEEP": "v3_cpu_final_verified.csv",
    "MEM16": "v3_mem15_warm.csv",
    "MEM23": "v3_mem25_warm.csv",
    "CACHE_WARM": "v3_cache_warm.csv",
    "CACHE_COLD": "v3_cache_cold_pertrial.csv",
    "COMBINED_CPU11_MEM16": "v3_combined_cpu11_mem16.csv",
    "COMBINED_HIGHCPU_MEM16": "v3_combined_cpu19_mem23.csv",
}

frames = []

for source_condition, filename in DATASETS.items():
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {filename}")

    df = pd.read_csv(path)

    # Never trust old hard-coded experiment labels.
    df["source_condition"] = source_condition

    # Preserve original label for traceability.
    if "experiment" in df.columns:
        df["experiment_original"] = df["experiment"]

    # Our authoritative experiment label.
    df["experiment"] = source_condition

    frames.append(df)

master = pd.concat(frames, ignore_index=True)

# Numeric versions for modeling.
for col in [
    "pre_scale_node_cpu_pct",
    "pre_scale_node_memory_pct",
    "request_to_ready_ms",
]:
    if col in master.columns:
        master[col] = (
            master[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        master[col] = pd.to_numeric(master[col], errors="coerce")

# Sort deterministically.
master = master.sort_values(
    ["source_condition", "trial"]
).reset_index(drop=True)

output = "v3_master_100_raw.csv"
master.to_csv(output, index=False)

print("=" * 60)
print("V3 MASTER DATASET CREATED")
print("=" * 60)
print(f"Rows   : {len(master)}")
print(f"Columns: {len(master.columns)}")
print(f"Output : {output}")

print("\nRows by source:")
print(master["source_condition"].value_counts().sort_index())

print("\nMissing values:")
print(master.isna().sum()[master.isna().sum() > 0])

print("\nLatency summary:")
print(master["request_to_ready_ms"].describe())

print("\nCPU range:")
print(master["pre_scale_node_cpu_pct"].min(),
      "to",
      master["pre_scale_node_cpu_pct"].max())

print("\nMemory range:")
print(master["pre_scale_node_memory_pct"].min(),
      "to",
      master["pre_scale_node_memory_pct"].max())