import pandas as pd
import numpy as np

INPUT = "azure_functions_aggregate_1min.parquet"
OUTPUT = "azure_replay_20min.csv"

REPLAY_SECONDS = 20 * 60

df = pd.read_parquet(INPUT)

df = df[["minute", "invocations"]].copy()

df["invocations"] = pd.to_numeric(
    df["invocations"],
    errors="coerce"
)

df = df.dropna().reset_index(drop=True)

# Original trace:
# 1440 one-minute observations
#
# Convert each minute's invocation count
# into requests/second.

df["requests_per_second"] = (
    df["invocations"] / 60.0
)

# Original trace positions
original_x = np.linspace(
    0,
    REPLAY_SECONDS - 1,
    len(df)
)

# Replay positions: one value per second
replay_x = np.arange(
    REPLAY_SECONDS
)

# Interpolate the real Azure trace
replay_rps = np.interp(
    replay_x,
    original_x,
    df["requests_per_second"].values
)

out = pd.DataFrame({

    "second": replay_x,

    "requests_per_second": replay_rps,

})

out["invocations_per_minute"] = (
    out["requests_per_second"] * 60.0
)

out.to_csv(
    OUTPUT,
    index=False
)

print("=" * 70)
print("STARTSCALE - AZURE TRACE REPLAY")
print("=" * 70)

print(f"Original rows : {len(df)}")
print(f"Replay seconds: {len(out)}")
print(f"Replay minutes: {REPLAY_SECONDS / 60:.1f}")

print()
print("Original statistics:")
print(f"Min  : {df['invocations'].min():,.0f}")
print(f"Max  : {df['invocations'].max():,.0f}")
print(f"Mean : {df['invocations'].mean():,.0f}")

print()
print("Replay RPS:")
print(f"Min  : {out['requests_per_second'].min():,.2f}")
print(f"Max  : {out['requests_per_second'].max():,.2f}")
print(f"Mean : {out['requests_per_second'].mean():,.2f}")

print()
print("First 10 replay points:")
print(out.head(10).to_string(index=False))

print()
print(f"Saved: {OUTPUT}")

