"""
Preprocess the Azure Functions 2019 invocation trace (Shahrad et al., "Serverless in the
Wild", USENIX ATC 2020 -- azurefunctions_dataset2019) for use as a "bursty / serverless
cold-start-relevant" traffic trace (guide: Datasets -> Borrowed traces, optional third
trace if you frame the paper toward FaaS).

WHY THIS DATASET / WHY THIS CODE
---------------------------------
- The public release ships several CSVs per day (d01..d14). The one you want for a
  traffic trace is invocations_per_function_md.anon.d<NN>.csv. Its schema is:

    HashOwner, HashApp, HashFunction, Trigger, 1, 2, 3, ..., 1440

  Each row is one function; columns "1".."1440" are invocation COUNTS for each minute
  of that day (1440 minutes/day). This is already an arrival-count trace, just organised
  per-function instead of per-cluster, so preprocessing here is mostly RESHAPING, not
  parsing text like the NASA/CLF logs.
- StartScale drives a SINGLE target deployment with Locust, so this script offers two
  aggregation modes:
    * "aggregate": sum invocation counts across every function into one cluster-level
      per-minute arrival trace -- gives you serverless-style burstiness without picking
      a specific function.
    * "top-function": pick the single busiest function and use its own per-minute trace
      -- more realistic if your paper frames this as "one FaaS-style endpoint", and lets
      you also report that function's HashApp/HashFunction for reproducibility.
  Both are computed so you can decide later; the "note" in the printed summary tells you
  which one had the higher peak-to-mean ratio (the more "bursty" one), since that's the
  property the guide actually wants from this trace.
- Minute-resolution is native to the dataset (no finer-grained timestamps exist), so the
  output trace is at 1-minute resolution -- upsample later if Locust needs finer steps,
  but do not invent sub-minute structure that was not in the source data.
- Trigger type is kept in the per-function breakdown because "http" vs "timer" vs "queue"
  triggers behave differently; filtering to HTTP-triggered functions before picking a
  "top function" is worth doing if your paper is about web-facing autoscaling.

OUTPUT
------
1. <name>_per_function.parquet  -- long-format table: HashApp, HashFunction, Trigger,
                                    minute (1..1440), invocations
2. <name>_aggregate_1min.parquet -- one row per minute, summed across all functions
3. <name>_top_function_1min.parquet -- one row per minute, for the single busiest function
4. Printed summary comparing the two candidate traces.

USAGE
-----
    python preprocess_azure_functions.py invocations_per_function_md.anon.d01.csv azure_d01
"""

import os
import sys

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pandas/pyarrow are not installed in this environment.\n"
        "Install them with:\n"
        "  c:/Users/ELCOT/Desktop/Web_Autoscaling/.venv/Scripts/python.exe -m pip install pandas pyarrow"
    ) from exc


def preprocess(input_csv: str, output_prefix: str):
    if not os.path.isfile(input_csv):
        print(f"Input CSV not found: {input_csv}")
        print(
            "Expected a file like 'invocations_per_function_md.anon.d01.csv'. "
            "Extract the Azure Functions dataset archive first or pass the correct CSV path."
        )
        return

    df = pd.read_csv(input_csv)

    minute_cols = [c for c in df.columns if c.isdigit()]
    if not minute_cols:
        print(
            "No numeric minute columns found. This script expects the "
            "'invocations_per_function_md.anon.dNN.csv' file specifically -- "
            "check you didn't point it at a durations/memory file instead."
        )
        return
    minute_cols_sorted = sorted(minute_cols, key=int)

    id_cols = [c for c in ["HashOwner", "HashApp", "HashFunction", "Trigger"] if c in df.columns]

    # 1. Long format: one row per (function, minute). Useful for later filtering by
    #    Trigger type or for per-function modelling, without re-parsing the wide CSV.
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=minute_cols_sorted,
        var_name="minute",
        value_name="invocations",
    )
    long_df["minute"] = long_df["minute"].astype(int)
    per_function_path = f"{output_prefix}_per_function.parquet"
    long_df.to_parquet(per_function_path, index=False)

    # 2. Cluster-level aggregate trace: sum every function's count per minute.
    agg = df[minute_cols_sorted].sum(axis=0)
    agg.index = agg.index.astype(int)
    agg = agg.sort_index()
    agg_df = agg.rename("invocations").reset_index().rename(columns={"index": "minute"})
    agg_path = f"{output_prefix}_aggregate_1min.parquet"
    agg_df.to_parquet(agg_path, index=False)

    # 3. Single busiest function's own trace -- total invocations across the day decides
    #    "busiest"; peak-to-mean ratio (computed below) decides "burstiest", which is
    #    reported separately since the two are not always the same function.
    totals = df[minute_cols_sorted].sum(axis=1)
    top_idx = totals.idxmax()
    top_row = df.loc[top_idx]
    top_counts = top_row[minute_cols_sorted]
    top_counts.index = top_counts.index.astype(int)
    top_counts = top_counts.sort_index()
    top_df = top_counts.rename("invocations").reset_index().rename(columns={"index": "minute"})
    top_path = f"{output_prefix}_top_function_1min.parquet"
    top_df.to_parquet(top_path, index=False)

    def peak_to_mean(series: pd.Series) -> float:
        m = series.mean()
        return float(series.max() / m) if m > 0 else float("inf")

    agg_ptm = peak_to_mean(agg_df["invocations"])
    top_ptm = peak_to_mean(top_df["invocations"])

    print(f"--- {input_csv} ---")
    print(f"Functions in file:        {len(df)}")
    print(f"Minute columns found:     {len(minute_cols_sorted)} (expect 1440 for a full day)")
    if id_cols and "HashApp" in top_row and "HashFunction" in top_row:
        print(f"Busiest function:         HashApp={top_row['HashApp']} HashFunction={top_row['HashFunction']}")
    if "Trigger" in top_row:
        print(f"Busiest function trigger: {top_row['Trigger']}")
    print(f"Aggregate trace peak/mean ratio:     {agg_ptm:.2f}")
    print(f"Top-function trace peak/mean ratio:  {top_ptm:.2f}")
    print(
        "-> "
        + (
            "Aggregate trace is burstier; consider using it."
            if agg_ptm >= top_ptm
            else "Top-function trace is burstier; consider using it if your paper frames "
            "this as a single FaaS endpoint."
        )
    )
    print(f"Wrote: {per_function_path}")
    print(f"Wrote: {agg_path}")
    print(f"Wrote: {top_path}")
    print(
        "\nNote: this trace is minute-resolution invocation COUNTS, not per-second arrival "
        "times. Convert to a Locust arrival-rate shape by treating each minute's count as "
        "a target rate for that minute (optionally interpolate between minutes), then "
        "rescale to your cluster's capacity as with the NASA trace."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_azure_functions.py <invocations_per_function_md.anon.dNN.csv> <output_prefix>")
        sys.exit(1)
    preprocess(sys.argv[1], sys.argv[2])
