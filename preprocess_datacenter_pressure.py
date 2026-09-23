"""
Preprocess the datacenter machine-usage traces (Alibaba2018 / Azure_v2 / Google2019,
bundled via the datacentertracesdatasets-cli repo) for use as REALISTIC NODE PRESSURE
INJECTION SHAPES -- not as a Locust request-arrival trace.

WHY THIS IS A DIFFERENT KIND OF DATASET THAN NASA / WORLDCUP98 / AZURE FUNCTIONS
-----------------------------------------------------------------------------------
The other three traces record HTTP/function REQUEST ARRIVALS -- exactly what Locust
replays to generate load. These CSVs record something else: whole-datacenter CPU and
memory UTILIZATION over time (cpu_util_percent, mem_util_percent, etc., at 300-second
resolution). There is no request count here at all.

Your guide's factorial sweep (Phase 3) has "Node CPU pressure: low/medium/high" and
"Node memory pressure: low/high" as FACTORS, currently meant to be produced by the
stress-ng DaemonSet at three fixed constant levels. That's fine for a controlled sweep,
but fixed constants are also exactly the kind of simplification the whole StartScale
thesis argues against for readiness latency -- so this dataset's real use is as a
REFERENCE for what pressure actually looks like over time in production, either to:
  (a) justify your chosen low/medium/high thresholds with a real-world citation, or
  (b) optionally replace the constant stress-ng levels with a realistic pressure
      SCHEDULE (ramping/bursty CPU load) sampled from this trace, for a more externally
      valid pressure factor -- if you choose to do this, treat it as a scope addition,
      not something the guide currently asks for.

This is NOT a drop-in replacement for the NASA/WorldCup98/Azure Functions traces, and it
does not get run through Locust.

WHY THE PREPROCESSING DIFFERS PER SOURCE
-------------------------------------------
The three CSVs use different units, discovered by inspecting the files directly rather
than assuming from the README:
  - google2019-*.csv / my_dataset.csv: already PERCENTAGES (0-100), columns
    cpu_util_percent, mem_util_percent, assigned_mem_percent, cycles_per_instruction.
  - azure_v2-*.csv: RAW, unbounded values (summed core-count usage and MB of assigned
    memory across the whole simulated datacenter) -- columns cpu_usage, assigned_mem.
    These must be min-max normalized to compare on the same [0, 1] scale as the
    Google trace; there is no natural denominator given in the source data, so the
    normalization is relative to this file's own observed range, not an absolute
    capacity. State that limitation if you cite this in the paper.
Both datasets are unlabeled with real timestamps (they are anonymized/aggregated
utilization series, not event logs), so this script assigns a synthetic 300-second
index starting at t=0 rather than inventing a fake calendar date.

WHAT THE SCRIPT PRODUCES
---------------------------
1. <name>_normalized.parquet -- timestamp (relative, 300s steps), cpu_util (0-1),
   mem_util (0-1, only if the source has a memory column)
2. Tertile-based low/medium/high (CPU) and median-split low/high (memory) labels,
   matching the guide's factor levels -- these are descriptive labels for reporting
   real-world pressure distribution, not a replacement for your controlled sweep design.
3. Printed summary: value ranges, label thresholds, distribution across levels.

USAGE
-----
    python preprocess_datacenter_pressure.py google2019-machine_usage-300s-synthetic-...csv google2019_pressure --schema percent
    python preprocess_datacenter_pressure.py azure_v2-machine_usage-300s-real-...csv azure_pressure --schema raw
"""

import argparse

import pandas as pd


def preprocess(input_csv: str, output_prefix: str, schema: str):
    df = pd.read_csv(input_csv)

    if schema == "percent":
        if "cpu_util_percent" not in df.columns:
            print("Expected a 'cpu_util_percent' column for schema=percent; check the file.")
            return
        cpu = df["cpu_util_percent"] / 100.0
        mem = df["mem_util_percent"] / 100.0 if "mem_util_percent" in df.columns else None
    elif schema == "raw":
        if "cpu_usage" not in df.columns:
            print("Expected a 'cpu_usage' column for schema=raw; check the file.")
            return
        # Min-max normalize within this file -- there is no absolute capacity figure
        # in the source data, so this is a RELATIVE scale, not a true utilization
        # percentage. Document this limitation wherever this trace is cited.
        cpu = (df["cpu_usage"] - df["cpu_usage"].min()) / (df["cpu_usage"].max() - df["cpu_usage"].min())
        mem = None
        if "assigned_mem" in df.columns:
            mem = (df["assigned_mem"] - df["assigned_mem"].min()) / (
                df["assigned_mem"].max() - df["assigned_mem"].min()
            )
    else:
        print("schema must be 'percent' or 'raw'")
        return

    out = pd.DataFrame({
        "timestamp_s": pd.RangeIndex(len(df)) * 300,  # synthetic relative clock, 300s steps
        "cpu_util": cpu.values,
    })
    if mem is not None:
        out["mem_util"] = mem.values

    # Tertile split for CPU (matches the guide's 3-level factor: low/medium/high).
    cpu_low_thr, cpu_high_thr = out["cpu_util"].quantile([1 / 3, 2 / 3])
    out["cpu_pressure_label"] = pd.cut(
        out["cpu_util"], bins=[-float("inf"), cpu_low_thr, cpu_high_thr, float("inf")],
        labels=["low", "medium", "high"],
    )

    mem_thr = None
    if "mem_util" in out.columns:
        # Median split for memory (matches the guide's 2-level factor: low/high).
        mem_thr = out["mem_util"].median()
        out["mem_pressure_label"] = pd.cut(
            out["mem_util"], bins=[-float("inf"), mem_thr, float("inf")], labels=["low", "high"],
        )

    out_path = f"{output_prefix}_normalized.parquet"
    out.to_parquet(out_path, index=False)

    print(f"--- {input_csv} (schema={schema}) ---")
    print(f"Rows (300s steps):     {len(out)} ({len(out) * 300 / 3600:.1f} hours)")
    print(f"CPU util range:        {out['cpu_util'].min():.3f} .. {out['cpu_util'].max():.3f}")
    print(f"CPU tertile thresholds: low<{cpu_low_thr:.3f}<=medium<{cpu_high_thr:.3f}<=high")
    print(out["cpu_pressure_label"].value_counts().sort_index())
    if "mem_util" in out.columns:
        print(f"Mem util range:        {out['mem_util'].min():.3f} .. {out['mem_util'].max():.3f}")
        print(f"Mem median threshold:  {mem_thr:.3f}")
        print(out["mem_pressure_label"].value_counts().sort_index())
    else:
        print("No memory column in this source -- mem_pressure_label not produced.")
    print(f"Wrote: {out_path}")
    print(
        "\nReminder: this trace has no bearing on request arrival rate or Locust replay. "
        "Its only role is characterizing/justifying node CPU and memory pressure levels."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_prefix")
    parser.add_argument("--schema", choices=["percent", "raw"], required=True,
                         help="'percent' for google2019/my_dataset CSVs, 'raw' for azure_v2 CSVs")
    args = parser.parse_args()
    preprocess(args.input_csv, args.output_prefix, args.schema)
