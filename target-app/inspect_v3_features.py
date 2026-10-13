import pandas as pd

FILE = "v3_master_100_raw.csv"

df = pd.read_csv(FILE)

print("=" * 80)
print("STARTSCALE V3 FEATURE INVENTORY")
print("=" * 80)

print("\n[1] ALL COLUMNS")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n[2] COLUMN TYPES")
print(df.dtypes.to_string())

print("\n[3] UNIQUE VALUE COUNTS")
for col in df.columns:
    print(
        f"{col:35s} "
        f"unique={df[col].nunique(dropna=False):3d} "
        f"missing={df[col].isna().sum():3d}"
    )

print("\n[4] CONSTANT COLUMNS")
constant_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

if constant_cols:
    for col in constant_cols:
        print("-", col)
else:
    print("None")

print("\n[5] POTENTIALLY USEFUL NUMERIC COLUMNS")

for col in df.columns:
    numeric = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    if numeric.notna().sum() >= 50:
        print(
            f"{col:35s}"
            f" min={numeric.min():.3f}"
            f" max={numeric.max():.3f}"
            f" unique={numeric.nunique()}"
        )

print("\n[6] IMPORTANT SAMPLE ROW")

print(
    df[
        [
            "pre_scale_node_cpu_pct",
            "pre_scale_node_memory_pct",
            "image_cache_state",
            "image",
            "image_id",
            "cpu_request",
            "memory_request",
            "configured_startup_delay_ms",
            "node_name",
            "request_to_ready_ms",
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 80)
print("FEATURE INVENTORY COMPLETE")
print("=" * 80)