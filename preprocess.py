"""
Preprocess the Azure Functions 2019 invocation trace (Shahrad et al., "Serverless in the
Wild", USENIX ATC 2020 -- azurefunctions_dataset2019) for use as a "bursty / serverless
cold-start-relevant" traffic trace (guide: Datasets -> Borrowed traces, optional third
trace if you frame the paper toward FaaS).

WHY THIS DATASET / WHY THIS CODE
---------------------------------
- Each daily CSV (d01..d14) has schema: HashOwner, HashApp, HashFunction, Trigger,
  then columns "1".."1440" -- per-minute invocation COUNTS for that day, one row per
  function. This is already an arrival-count trace, just organised per-function instead
  of per-cluster, so preprocessing here is RESHAPING, not text parsing.
- StartScale drives a SINGLE target deployment with Locust, so only two outputs matter:
    * "aggregate": sum invocation counts across every function -> one cluster-level
      per-minute arrival trace, serverless-style burstiness without picking a function.
    * "top-function": the single busiest function's own per-minute trace -- more
      realistic if you frame this as one FaaS-style endpoint.
  Both are cheap: they only need a column-wise sum over the wide CSV, never a reshape.
- Minute-resolution is native to the dataset -- upsample later if Locust needs finer
  steps, but do not invent sub-minute structure that wasn't in the source data.

WHY THE LONG-FORMAT (PER-FUNCTION-PER-MINUTE) TABLE IS *NOT* WRITTEN BY DEFAULT
---------------------------------------------------------------------------------
An earlier version of this script also melted the wide CSV into one row per
(function, minute) -- rows = num_functions * 1440. On a real daily file (tens of
thousands of functions), that is tens of millions of long-format rows, which can run
a multi-GB Parquet write and exhausted disk space on a real run ("No space left on
device"). You don't need that table for anything in the StartScale pipeline -- the
aggregate and top-function traces above are computed directly from the wide CSV without
ever building it. It's now behind an explicit --long-format flag, off by default, with
a printed size estimate so you can judge free disk space before turning it on.

OUTPUT (default)
-----------------
1. <name>_aggregate_1min.parquet    -- one row per minute, summed across all functions
2. <name>_top_function_1min.parquet -- one row per minute, for the single busiest function
3. Printed summary comparing the two candidate traces.

OUTPUT (only with --long-format)
---------------------------------
4. <name>_per_function.parquet -- long format: HashApp, HashFunction, Trigger, minute,
   invocations. Can be tens of millions of rows -- check the printed size estimate first.

USAGE
-----
    python preprocess_azure_functions.py invocations_per_function_md.anon.d01.csv azure_d01
    python preprocess_azure_functions.py invocations_per_function_md.anon.d01.csv azure_d01 --long-format
"""

import argparse

import pandas as pd


def preprocess(input_csv: str, output_prefix: str, long_format: bool):
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

    # Cluster-level aggregate trace: sum every function's count per minute.
    # Column-wise sum over the wide CSV -- no reshape, cheap regardless of row count.
    agg = df[minute_cols_sorted].sum(axis=0)
    agg.index = agg.index.astype(int)
    agg = agg.sort_index()
    agg_df = agg.rename("invocations").reset_index().rename(columns={"index": "minute"})
    agg_path = f"{output_prefix}_aggregate_1min.parquet"
    agg_df.to_parquet(agg_path, index=False)

    # Single busiest function's own trace -- total invocations across the day decides
    # "busiest"; peak-to-mean ratio decides "burstiest", reported separately since the
    # two are not always the same function.
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

    est_long_rows = len(df) * len(minute_cols_sorted)

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
    print(f"Wrote: {agg_path}")
    print(f"Wrote: {top_path}")

    if long_format:
        print(
            f"\n--long-format requested: melting to ~{est_long_rows:,} rows "
            f"({len(df):,} functions x {len(minute_cols_sorted)} minutes). This can take "
            "a while and use several GB of disk -- interrupt now (Ctrl+C) if that's a "
            "problem."
        )
        long_df = df.melt(
            id_vars=id_cols,
            value_vars=minute_cols_sorted,
            var_name="minute",
            value_name="invocations",
        )
        long_df["minute"] = long_df["minute"].astype(int)
        per_function_path = f"{output_prefix}_per_function.parquet"
        long_df.to_parquet(per_function_path, index=False)
        print(f"Wrote: {per_function_path} ({len(long_df):,} rows)")
    else:
        print(
            f"\nSkipped the long-format per-function table (~{est_long_rows:,} rows if "
            "built) -- not needed for the aggregate/top-function traces above. Pass "
            "--long-format if you specifically need per-function-per-minute rows."
        )

    print(
        "\nNote: this trace is minute-resolution invocation COUNTS, not per-second arrival "
        "times. Convert to a Locust arrival-rate shape by treating each minute's count as "
        "a target rate for that minute (optionally interpolate between minutes), then "
        "rescale to your cluster's capacity as with the NASA trace."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to invocations_per_function_md.anon.dNN.csv")
    parser.add_argument("output_prefix")
    parser.add_argument(
        "--long-format", action="store_true",
        help="Also write the per-function-per-minute long table (can be tens of millions of rows / several GB)",
    )
    args = parser.parse_args()
    preprocess(args.input_csv, args.output_prefix, args.long_format)