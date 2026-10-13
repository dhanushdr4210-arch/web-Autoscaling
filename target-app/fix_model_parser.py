from pathlib import Path

path = Path("model_v2_140.py")
text = path.read_text()

old = """        cpu_pct = parse_percent(
            node_data.get("cpu_pct")
        )

        memory_pct = parse_percent(
            node_data.get("memory_pct")
        )"""

new = """        cpu_m = pd.to_numeric(
            node_data.get("cpu_m"),
            errors="coerce"
        )
        cpu_pct = (
            (cpu_m / 16000.0) * 100.0
            if not pd.isna(cpu_m)
            else np.nan
        )

        memory_mi = pd.to_numeric(
            node_data.get("memory_mi"),
            errors="coerce"
        )
        memory_pct = (
            (memory_mi / (7970460 / 1024.0)) * 100.0
            if not pd.isna(memory_mi)
            else np.nan
        )"""

if old not in text:
    raise RuntimeError("Target code block was not found. No changes made.")

path.write_text(text.replace(old, new, 1))

print("PATCH APPLIED SUCCESSFULLY")