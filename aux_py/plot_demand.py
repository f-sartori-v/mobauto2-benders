import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

yaml_path = PROJECT_DIR / "setups" / "prob" / "base_plus100_out_noon.yaml"
csv_path = SCRIPT_DIR / f"{yaml_path.stem}__direction_counts.csv"
png_path = SCRIPT_DIR / f"{yaml_path.stem}_demand.png"

with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

requests = pd.DataFrame(data["requests"])
requests["time_bin"] = (requests["time"] // 30) * 30

summary = (
    requests.groupby(["time_bin", "dir"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
    .sort_values("time_bin")
)

for col in ["OUT", "RET"]:
    if col not in summary.columns:
        summary[col] = 0

base_minutes = 7 * 60
summary["clock_time"] = summary["time_bin"].apply(
    lambda m: f"{(base_minutes + m)//60}:{(base_minutes + m)%60:02d}"
)

summary[["time_bin", "clock_time", "OUT", "RET"]].to_csv(csv_path, index=False)

x = range(len(summary))
width = 0.4

plt.figure(figsize=(12, 6))
plt.bar([i - width/2 for i in x], summary["OUT"], width=width, label="to-Hub")
plt.bar([i + width/2 for i in x], summary["RET"], width=width, label="from-Hub")

plt.xticks(list(x), summary["clock_time"], rotation=0)
plt.xlabel("Time")
plt.ylim(0, 30)
plt.yticks(range(0, 31, 5))
plt.ylabel("Number of bookings")
plt.title("Demand by time slot and direction")
plt.legend()
plt.tight_layout()

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()

summary[["clock_time", "OUT", "RET"]]