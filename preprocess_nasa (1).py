"""
Preprocess the NASA-HTTP access logs (NASA_access_log_Jul95.gz, NASA_access_log_Aug95.gz)
for use as a "diurnal" traffic trace in StartScale's Locust replay (guide: Datasets ->
Borrowed: traffic traces to drive load).

WHY THIS DATASET / WHY THIS CODE
---------------------------------
- NASA-HTTP is a classic web-server access log with a clear day/night cycle, which is
  exactly the "baseline and gradual-ramp scenarios" role the guide assigns it.
- The files are gzip-compressed Common Log Format (CLF) text, one line per HTTP request:

    host - - [DD/Mon/YYYY:HH:MM:SS -0000] "METHOD path HTTP/x.x" status bytes

  e.g.  199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245

  CLF is a fixed, well-documented format (no external lookup tables needed), so it can be
  parsed directly with a regex -- unlike the WorldCup98 trace, which uses a custom binary
  encoding (see the README in this folder for why that one is NOT included yet).
- We only need REQUEST ARRIVAL TIMES to drive Locust, not the payload content, so the
  pipeline reduces each line to (timestamp, method, path, status, bytes) and then throws
  away everything except timestamp for the trace itself. The other fields are kept in the
  Parquet output in case you want to filter (e.g. only GET, only 200s) before resampling.
- We stream the gzip file line-by-line instead of loading it fully into memory. The files
  are small enough (~20 MB compressed) that this isn't strictly required, but it's the
  pattern you'll want later for the multi-GB Alibaba/Google traces, so it's used
  consistently here.
- Malformed lines are counted, not silently dropped without a trace -- the guide's
  reproducibility bar expects you to report data-quality issues like this in the paper.

OUTPUT
------
1. <name>_requests.parquet   -- one row per parsed request (timestamp, method, path, status, bytes)
2. <name>_arrivals_1s.parquet -- requests-per-second time series (for Locust's custom shape)
3. A short printed summary: line count, parse failures, date range, peak/median RPS.

USAGE
-----
    python preprocess_nasa.py NASA_access_log_Jul95.gz nasa_jul95
    python preprocess_nasa.py NASA_access_log_Aug95.gz nasa_aug95
"""

import gzip
import re
import sys
from datetime import datetime

import pandas as pd

# CLF regex. Groups: host, timestamp, method, path, protocol, status, bytes.
# `bytes` is sometimes "-" (no content), handled below.
CLF_RE = re.compile(
    r'^(?P<host>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] '
    r'"(?P<method>\S+)?\s?(?P<path>\S+)?\s?(?P<protocol>[^"]*)" '
    r'(?P<status>\d{3}|-) (?P<bytes>\d+|-)\s*$'
)

# CLF timestamp format, e.g. "01/Jul/1995:00:00:01 -0400"
TS_FORMAT = "%d/%b/%Y:%H:%M:%S %z"


def parse_line(line: str):
    m = CLF_RE.match(line)
    if not m:
        return None
    d = m.groupdict()
    try:
        ts = datetime.strptime(d["ts"], TS_FORMAT)
    except ValueError:
        return None
    return {
        "timestamp": ts,
        "host": d["host"],
        "method": d["method"] or None,
        "path": d["path"] or None,
        "status": int(d["status"]) if d["status"].isdigit() else None,
        "bytes": int(d["bytes"]) if d["bytes"].isdigit() else 0,
    }


def preprocess(input_path: str, output_prefix: str):
    total = 0
    failed = 0
    rows = []

    opener = gzip.open if input_path.endswith(".gz") else open
    with opener(input_path, "rt", encoding="latin-1", errors="replace") as f:
        for line in f:
            total += 1
            rec = parse_line(line)
            if rec is None:
                failed += 1
                continue
            rows.append(rec)

    if not rows:
        print(f"No rows parsed from {input_path}. Check the file format before proceeding.")
        return

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # 1. Per-request table -- kept for later filtering (e.g. by status code or path).
    requests_path = f"{output_prefix}_requests.parquet"
    df.to_parquet(requests_path, index=False)

    # 2. Arrival-rate trace at 1-second resolution -- this is what Locust replays.
    #    Reindexed so seconds with zero requests are explicit 0s, not missing rows;
    #    a gap in the index would otherwise look like "no data" rather than "no traffic".
    counts = (
        df.set_index("timestamp")
        .resample("1s")
        .size()
        .rename("requests_per_second")
    )
    full_index = pd.date_range(counts.index.min(), counts.index.max(), freq="1s")
    counts = counts.reindex(full_index, fill_value=0)
    counts.index.name = "timestamp"
    arrivals_path = f"{output_prefix}_arrivals_1s.parquet"
    counts.reset_index().to_parquet(arrivals_path, index=False)

    print(f"--- {input_path} ---")
    print(f"Lines read:        {total}")
    print(f"Parse failures:    {failed} ({failed / total:.2%})")
    print(f"Requests parsed:   {len(df)}")
    print(f"Date range:        {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"Peak RPS:          {counts.max()}")
    print(f"Median RPS:        {counts.median()}")
    print(f"Mean RPS:          {counts.mean():.2f}")
    print(f"Wrote:             {requests_path}")
    print(f"Wrote:             {arrivals_path}")
    print(
        "\nNote: rescale requests_per_second to your test cluster's actual capacity "
        "before replaying (e.g. target_rps = requests_per_second * scale_factor). "
        "The guide requires you to state the scale_factor and rationale in the paper."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_nasa.py <input.gz> <output_prefix>")
        sys.exit(1)
    preprocess(sys.argv[1], sys.argv[2])
