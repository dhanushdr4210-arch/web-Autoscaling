import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu


warm_file = "startscale_10_trials.csv"
cold_file = "startscale_10_cold_trials.csv"

warm = pd.read_csv(warm_file)
cold = pd.read_csv(cold_file)

warm_values = warm.loc[
    warm["status"] == "SUCCESS",
    "request_to_ready_ms"
].astype(float)

cold_values = cold.loc[
    cold["status"] == "SUCCESS",
    "request_to_ready_ms"
].astype(float)


print("=" * 80)
print("STARTSCALE CACHE EFFECT ANALYSIS")
print("=" * 80)

print()
print("WARM CACHE")
print("-" * 80)
print(f"N       : {len(warm_values)}")
print(f"Mean    : {warm_values.mean():.2f} ms")
print(f"Median  : {warm_values.median():.2f} ms")
print(f"Std     : {warm_values.std(ddof=1):.2f} ms")
print(f"Min     : {warm_values.min():.2f} ms")
print(f"Max     : {warm_values.max():.2f} ms")

print()
print("COLD CACHE")
print("-" * 80)
print(f"N       : {len(cold_values)}")
print(f"Mean    : {cold_values.mean():.2f} ms")
print(f"Median  : {cold_values.median():.2f} ms")
print(f"Std     : {cold_values.std(ddof=1):.2f} ms")
print(f"Min     : {cold_values.min():.2f} ms")
print(f"Max     : {cold_values.max():.2f} ms")


mean_difference = cold_values.mean() - warm_values.mean()
percentage_difference = (
    mean_difference / warm_values.mean()
) * 100


print()
print("CACHE EFFECT")
print("-" * 80)
print(f"Mean difference  : {mean_difference:.2f} ms")
print(f"Percentage       : {percentage_difference:.2f}%")


# Welch's t-test
t_stat, t_p = ttest_ind(
    warm_values,
    cold_values,
    equal_var=False
)

print()
print("WELCH'S T-TEST")
print("-" * 80)
print(f"t-statistic : {t_stat:.4f}")
print(f"p-value     : {t_p:.6f}")


# Mann-Whitney U test
u_stat, u_p = mannwhitneyu(
    warm_values,
    cold_values,
    alternative="two-sided"
)

print()
print("MANN-WHITNEY U TEST")
print("-" * 80)
print(f"U-statistic : {u_stat:.4f}")
print(f"p-value     : {u_p:.6f}")


print()
print("INTERPRETATION")
print("-" * 80)

if t_p < 0.05:
    print(
        "Welch's test indicates a statistically significant "
        "difference at alpha = 0.05."
    )
else:
    print(
        "Welch's test does not establish significance at "
        "alpha = 0.05."
    )

if u_p < 0.05:
    print(
        "Mann-Whitney test also indicates a statistically "
        "significant distributional difference."
    )
else:
    print(
        "Mann-Whitney test does not establish significance "
        "at alpha = 0.05."
    )

print()