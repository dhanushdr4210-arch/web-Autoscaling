"""
Preprocess the WorldCup98 dataset for use as the "bursty" traffic trace in StartScale's
Locust replay (guide: Datasets -> Borrowed: FIFA World Cup 1998 web logs, "extreme bursts
around match times").

WHY THIS CODE, GIVEN WHAT'S ACTUALLY IN YOUR ZIP
--------------------------------------------------
The raw 1998 World Cup logs (from ita.ee.lbl.gov) are NOT plain text -- they use a custom,
fixed-width BINARY record format (timestamp offset, hashed client ID, hashed object ID,
size, packed method/status byte, packed type byte) with separate lookup tables to decode
the hashed fields. Writing that binary parser from scratch is error-prone: a wrong byte
width or endianness parses without crashing but produces silently wrong timestamps.

Your `worldcup98-dataset-master.zip` avoids all of that: it is a community repo that
already ran the real decoder (`ita_public_tools`, the official C reference implementation
for this exact format -- see `ita_public_tools/src/read.c`) over the raw logs and wrote the
result to `invocation_count.csv`, a plain two-column CSV:

    period,count            (period = minute timestamp, count = requests in that minute)

So there is no binary parsing left to do here -- this script's job is narrower: reshape that
already-correct per-minute trace into the same format as the NASA/Azure outputs, make gaps
explicit, and cut out a specific burst window instead of shipping all ~3 months at once
(replaying 3 months of traffic through Locust isn't the point; a few hours around a real
burst is).

WHY EXTRACT A WINDOW INSTEAD OF USING THE WHOLE FILE
------------------------------------------------------
The guide wants "spike and repeated-burst scenarios" -- a controlled few-hour segment you
can replay repeatedly, not an arbitrary 3-month span. This script:
  1. Loads the full per-minute series.
  2. Reindexes it to a complete minute-by-minute index (the source has ~10 missing minutes
     out of ~125,300 -- filled with 0 rather than left as gaps, since a gap would otherwise
     look like "no data" instead of "no traffic").
  3. Reports the single busiest minute and the busiest calendar day, so you can pick a
     real burst window instead of guessing at "the final" -- the actual traffic peak in
     this dataset does not necessarily fall on the final match date.
  4. Cuts out a configurable window (default: 6 hours centred on the busiest minute) as
     the actual trace to replay, in addition to writing the full trace.

OUTPUT
------
1. <name>_full_1min.parquet    -- the complete per-minute trace, gap-filled
2. <name>_burst_window.parquet -- the extracted high-traffic window (for the "single spike"
                                   / "repeated bursts" synthetic-adjacent scenarios)
3. Printed summary: date range, busiest minute, busiest day, chosen window.

USAGE
-----
    python preprocess_worldcup98.py invocation_count.csv worldcup98 --window-hours 6
"""

import argparse

import pandas as pd


def preprocess(input_csv: str, output_prefix: str, window_hours: int):
    df = pd.read_csv(input_csv, index_col=0, parse_dates=True).sort_index()
    df = df.rename(columns={df.columns[0]: "requests_per_minute"})

    # Fill the handful of missing minutes with 0 -- an absent row here means "no
    # requests logged," not "unknown," per the source repo's own convention.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    df = df.reindex(full_index, fill_value=0)
    df.index.name = "timestamp"

    full_path = f"{output_prefix}_full_1min.parquet"
    df.reset_index().to_parquet(full_path, index=False)

    busiest_minute = df["requests_per_minute"].idxmax()
    busiest_minute_value = df["requests_per_minute"].max()
    daily = df.resample("1D").sum()
    busiest_day = daily["requests_per_minute"].idxmax()

    # Window centred on the single busiest minute -- a real, data-driven burst rather
    # than an assumed date (e.g. the final), since the actual traffic peak may fall on
    # an earlier high-stakes match, a ticket-sales spike, or similar.
    half = pd.Timedelta(hours=window_hours / 2)
    window_start = busiest_minute - half
    window_end = busiest_minute + half
    window_df = df.loc[window_start:window_end]
    window_path = f"{output_prefix}_burst_window.parquet"
    window_df.reset_index().to_parquet(window_path, index=False)

    print(f"--- {input_csv} ---")
    print(f"Full date range:      {df.index.min()} .. {df.index.max()}")
    print(f"Total minutes:        {len(df)}")
    print(f"Busiest minute:       {busiest_minute} ({busiest_minute_value} requests)")
    print(f"Busiest day (sum):    {busiest_day.date()} ({daily.loc[busiest_day, 'requests_per_minute']} requests)")
    print(f"Burst window:         {window_start} .. {window_end} ({window_hours}h, {len(window_df)} minutes)")
    print(f"Window peak/mean:     {window_df['requests_per_minute'].max() / max(window_df['requests_per_minute'].mean(), 1e-9):.2f}")
    print(f"Wrote: {full_path}")
    print(f"Wrote: {window_path}")
    print(
        "\nNote: this is a per-MINUTE count, coarser than the NASA per-second trace. "
        "For Locust, treat each minute's count as the target rate for that minute "
        "(interpolate between minutes if you need finer steps), then rescale to your "
        "cluster's capacity as with the other traces. Check --window-hours against your "
        "actual experiment duration before finalising."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to invocation_count.csv from the worldcup98-dataset-master repo")
    parser.add_argument("output_prefix")
    parser.add_argument("--window-hours", type=int, default=6, help="Width of the extracted burst window (default: 6)")
    args = parser.parse_args()
    preprocess(args.input_csv, args.output_prefix, args.window_hours)
