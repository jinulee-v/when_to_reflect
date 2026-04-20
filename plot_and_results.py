import json
from collections import defaultdict, Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Make sure matplotlib doesn't try to parse LaTeX in labels
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.default"] = "regular"

# ==== INPUT FILE ====
# If you change model/filename later, just update this PATH.
PATH = "data/insert_reflection_math500_Qwen1.5-1.8B.jsonl"

# ==== LOAD RECORDS ====
records = []
with open(PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(records)} records from {PATH}")

# ==== AGGREGATE CONSISTENCY + ANSWER DISTRIBUTIONS ====

all_positions = []          # insertion positions
all_consistency = []        # max(answer_mass) / total_mass for that insert
answer_freq_by_pos = defaultdict(lambda: defaultdict(float))
global_answer_mass = Counter()

for obj in records:
    # critical token index (fallback position if per-insert position is missing)
    ct_idx = None
    if "critical_token_result" in obj:
        ct_idx = obj["critical_token_result"].get("critical_token_index", None)

    inserts = obj.get("insert_reflection_result", {}).get("and", [])
    if not isinstance(inserts, list):
        continue

    for ins in inserts:
        # Try a few reasonable key names for the insertion position
        pos = (
            ins.get("insert_index")
            or ins.get("insertion_index")
            or ins.get("position")
            or ct_idx
        )
        if pos is None:
            continue

        # Get answer distribution (probabilities or counts)
        dist = (
            ins.get("answer_distribution")
            or ins.get("answer")
            or {}
        )
        if not isinstance(dist, dict) or not dist:
            continue

        # Ensure numeric values
        try:
            total = float(sum(dist.values()))
        except Exception:
            continue

        if total <= 0:
            continue

        # Consistency for this insert: max(answer_mass) / total_mass
        max_freq = max(dist.values()) / total
        all_positions.append(pos)
        all_consistency.append(max_freq)

        # Accumulate frequencies by position and globally
        for ans, val in dist.items():
            try:
                v = float(val)
            except Exception:
                continue
            ans_str = str(ans)
            answer_freq_by_pos[pos][ans_str] += v
            global_answer_mass[ans_str] += v

# If we somehow got nothing, bail gracefully.
if not all_positions:
    print("No valid positions/answers found; nothing to plot.")
    raise SystemExit(0)

# ==== PLOT 1: CONSISTENCY VS POSITION ====

# Aggregate consistency by position (average over inserts at same pos)
consistency_by_pos = defaultdict(list)
for p, c in zip(all_positions, all_consistency):
    consistency_by_pos[p].append(c)

positions_sorted = sorted(consistency_by_pos.keys())
avg_consistency = [np.mean(consistency_by_pos[p]) for p in positions_sorted]

plt.figure(figsize=(8, 4))
plt.plot(positions_sorted, avg_consistency, marker="o", linewidth=1)
plt.xlabel("Insertion position (critical token index)")
plt.ylabel("Average consistency (max answer mass)")
plt.title("Consistency vs insertion position for 'and' (Qwen1.5-1.8B, math500)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_and_consistency.png", dpi=200)
plt.close()

# ==== PLOT 2: ANSWER DISTRIBUTION VS POSITION (STACKED AREA) ====

# Choose top-K answers globally; group rest as "Other"
TOP_K = 6
most_common_answers = [a for a, _ in global_answer_mass.most_common(TOP_K)]
answer_labels = most_common_answers + ["Other"]

# For each position, build normalized distribution over these labels
pos_probs = {p: {label: 0.0 for label in answer_labels} for p in positions_sorted}

for p in positions_sorted:
    dist = answer_freq_by_pos[p]
    total = sum(dist.values())
    if total <= 0:
        continue
    for ans, v in dist.items():
        if ans in most_common_answers:
            pos_probs[p][ans] += v / total
        else:
            pos_probs[p]["Other"] += v / total

# Build arrays for stackplot
y_by_label = {label: [] for label in answer_labels}
for p in positions_sorted:
    probs = pos_probs[p]
    for label in answer_labels:
        y_by_label[label].append(probs.get(label, 0.0))

plt.figure(figsize=(8, 4))
ys = [y_by_label[label] for label in answer_labels]

# No legend: avoids mathtext parsing hell with crazy LaTeX-like answer strings
plt.stackplot(positions_sorted, *ys)
plt.xlabel("Insertion position (critical token index)")
plt.ylabel("Answer probability")
plt.title("Answer distribution vs insertion position for 'and' (Qwen1.5-1.8B, math500)")
plt.tight_layout()
plt.savefig("plot_and_answer_distributions.png", dpi=200)
plt.close()

# ==== TEXT SUMMARY ====

summary_lines = []
summary_lines.append(f"Source file: {PATH}")
summary_lines.append(f"Number of records: {len(records)}")
summary_lines.append(f"Distinct insertion positions: {len(positions_sorted)}")
summary_lines.append(f"Distinct answers (raw): {len(global_answer_mass)}")
summary_lines.append("")
summary_lines.append("Top answers by total mass:")
for ans, mass in global_answer_mass.most_common(10):
    summary_lines.append(f"  {ans}: {mass:.3f}")
summary_lines.append("")
summary_lines.append(
    "Note: Consistency is defined as max(answer_mass) / total_mass for each insert.\n"
    "      Answer distributions are aggregated over positions and truncated to "
    f"top {TOP_K} answers plus 'Other'."
)

with open("and_results_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("Saved plot_and_consistency.png, plot_and_answer_distributions.png, and and_results_summary.txt")
