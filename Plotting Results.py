import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('AL_V3_Adding_Extra_Data.pkl', 'rb') as f:
    reports = pickle.load(f)

# === Metric Extraction and Plotting General ===
metrics = ["precision", "recall", "f1-score"]
target = "weighted avg"  # could be 'macro avg' or a class label like '0'

metric_values = {metric: [] for metric in metrics}
for report in reports:
    for metric in metrics:
        metric_values[metric].append(report[target][metric])

# Access the last report
last_report = reports[-1]

# Print the desired metrics
print("Metrics for the last report:")
for metric in metrics:
    value = last_report[target][metric]
    print(f"{metric.title()}: {value:.4f}")

for metric in metrics:
    plt.figure()
    plt.plot(range(1, len(metric_values[metric]) + 1), metric_values[metric], marker='o')
    plt.title(f"{metric.title()} Over Active Learning Rounds")
    plt.xlabel("Active Learning Round")
    plt.ylabel(metric.title())
    plt.xticks(np.arange(1, 20, 1))  # x-axis: 1 to 19
    plt.yticks(np.arange(0.0, 1.01, 0.1))  # y-axis: 0.0 to 1.0
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/V3/General/{metric}_over_rounds.png")
    # plt.show()


# === Metric Extraction and Plotting Separate ===
# Collect all keys from the first report (assumed consistent across all)
all_targets = reports[0].keys()
metrics = ["precision", "recall", "f1-score"]

# Loop through each target label (e.g. 'Auflage', 'macro avg', etc.)
for target in all_targets:
    metric_values = {metric: [] for metric in metrics}

    for report in reports:
        if target in report:  # in case some targets are missing in some rounds
            for metric in metrics:
                metric_values[metric].append(report[target][metric])
        else:
            for metric in metrics:
                metric_values[metric].append(float('nan'))  # fill missing with NaN

    # Plotting
    plt.figure()
    for metric in metrics:
        plt.plot(metric_values[metric], marker='o', label=metric.title())

    plt.title(f"Metrics for '{target}' Over Active Learning Rounds")
    plt.xlabel("Active Learning Round")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save each plot with a sanitized filename
    sanitized_name = target.replace(" ", "_").replace(":", "_")
    plt.savefig(f"plots/V3/Separate/metrics_{sanitized_name}.png")
    plt.close()

print("Done")