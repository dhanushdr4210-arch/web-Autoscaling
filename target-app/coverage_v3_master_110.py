import pandas as pd
from pathlib import Path


FILE = Path("v3_master_110_raw.csv")


def pct(series):
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )


def cpu_bin(x):
    if 0 <= x <= 2:
        return "CPU_0_2"
    elif 6 <= x <= 8:
        return "CPU_6_8"
    elif 11 <= x <= 14:
        return "CPU_11_14"
    elif 17 <= x <= 20:
        return "CPU_17_20"
    return "CPU_OTHER"


def mem_bin(x):
    if 2 <= x <= 4:
        return "MEM_2_4"
    elif 16 <= x <= 18:
        return "MEM_16_18"
    elif 22 <= x <= 23:
        return "MEM_22_23"
    return "MEM_OTHER"


def main():
    print("=" * 70)
    print(" StartScale V3 - 110 Row Coverage Analysis")
    print("=" * 70)

    df = pd.read_csv(FILE)

    df["cpu_pct_num"] = pct(
        df["pre_scale_node_cpu_pct"]
    )

    df["mem_pct_num"] = pct(
        df["pre_scale_node_memory_pct"]
    )

    df["cpu_bin"] = df["cpu_pct_num"].apply(cpu_bin)
    df["mem_bin"] = df["mem_pct_num"].apply(mem_bin)

    valid = df[
        (df["cpu_bin"] != "CPU_OTHER")
        & (df["mem_bin"] != "MEM_OTHER")
    ].copy()

    matrix = pd.crosstab(
        valid["cpu_bin"],
        valid["mem_bin"],
    )

    cpu_order = [
        "CPU_0_2",
        "CPU_6_8",
        "CPU_11_14",
        "CPU_17_20",
    ]

    mem_order = [
        "MEM_2_4",
        "MEM_16_18",
        "MEM_22_23",
    ]

    matrix = matrix.reindex(
        index=cpu_order,
        columns=mem_order,
        fill_value=0,
    )

    print()
    print("CPU x Memory coverage:")
    print()
    print(matrix.to_string())

    print()
    print("Occupied CPU x Memory cells:")

    occupied = 0
    missing = 0

    for cpu in cpu_order:
        for mem in mem_order:
            count = int(matrix.loc[cpu, mem])

            if count > 0:
                occupied += 1
                status = "FILLED"
            else:
                missing += 1
                status = "MISSING"

            print(
                f"  {cpu:12s} + {mem:11s} "
                f"= {count:2d} rows [{status}]"
            )

    print()
    print(f"Occupied cells : {occupied}/12")
    print(f"Missing cells   : {missing}/12")

    print()
    print("Missing cells:")

    for cpu in cpu_order:
        for mem in mem_order:
            if matrix.loc[cpu, mem] == 0:
                print(
                    f"  {cpu} + {mem}"
                )

    # ---------------------------------------------------------
    # Cache-aware exact combination coverage
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print(" Cache-aware coverage")
    print("=" * 70)

    exact = (
        valid.groupby(
            [
                "cpu_bin",
                "mem_bin",
                "image_cache_state",
            ]
        )
        .size()
    )

    for cpu in cpu_order:
        for mem in mem_order:
            warm = int(
                exact.get(
                    (cpu, mem, "warm"),
                    0,
                )
            )

            cold = int(
                exact.get(
                    (cpu, mem, "cold"),
                    0,
                )
            )

            if warm or cold:
                print(
                    f"{cpu:12s} + {mem:11s} "
                    f"-> warm={warm:2d}, cold={cold:2d}"
                )

    print()
    print("=" * 70)
    print(" COVERAGE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()