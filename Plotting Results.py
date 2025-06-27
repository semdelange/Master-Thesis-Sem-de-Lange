import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('AL_V2_Metrics_ResetModel_EntropyUncertainty.pkl', 'rb') as f:
    reports = pickle.load(f)

# # === Metric Extraction and Plotting General ===
# metrics = ["precision", "recall", "f1-score"]
# target = "weighted avg"  # could be 'macro avg' or a class label like '0'
#
# metric_values = {metric: [] for metric in metrics}
# for report in reports:
#     for metric in metrics:
#         metric_values[metric].append(report[target][metric])
#
# # Access the last report
# last_report = reports[0]
#
# # Print the desired metrics
# print("Metrics for the last report:")
# for metric in metrics:
#     value = last_report[target][metric]
#     print(f"{metric.title()}: {value:.4f}")
#
# # for metric in metrics:
# #     plt.figure()
# #     plt.plot(range(1, len(metric_values[metric]) + 1), metric_values[metric], marker='o')
# #     plt.title(f"{metric.title()} Over Active Learning Rounds")
# #     plt.xlabel("Active Learning Round")
# #     plt.ylabel(metric.title())
# #     plt.xticks(np.arange(1, 76, 1))  # x-axis: 1 to 76
# #     plt.yticks(np.arange(0.0, 1.01, 0.1))  # y-axis: 0.0 to 1.0
# #     plt.ylim(0.0, 1.0)
# #     plt.grid(True)
# #     plt.tight_layout()
# #     plt.savefig(f"plots/V5/General/{metric}_over_rounds.png")
# #     # plt.show()
#
# # Plot all metrics in one figure
# plt.figure(figsize=(10, 6))  # Optional: adjust size
# for metric in metrics:
#     plt.plot(
#         range(1, len(metric_values[metric]) + 1),
#         metric_values[metric],
#         marker='o',
#         label=metric.title()
#     )
#
# plt.title("Metrics Over Active Learning Rounds")
# plt.xlabel("Active Learning Round")
# plt.ylabel("Score")
# plt.xticks(np.arange(1, len(reports) + 1, 1))
# plt.yticks(np.arange(0.0, 1.01, 0.1))
# plt.ylim(0.0, 1.0)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("plots/V2/Together/V2_metrics_over_rounds.png")
# # plt.show()



# === Metric Extraction and Plotting General (excluding 'O') ===
metrics = ["precision", "recall", "f1-score"]
excluded_labels = ["O", "accuracy", "macro avg", "weighted avg"]

metric_values = {metric: [] for metric in metrics}

# Compute unweighted averages excluding label 'O'
for report in reports:
    # Initialize sums and count per report
    metric_sums = {metric: 0.0 for metric in metrics}
    count = 0
    for label, scores in report.items():
        if label not in excluded_labels:
            count += 1
            for metric in metrics:
                metric_sums[metric] += scores[metric]
    for metric in metrics:
        avg_value = metric_sums[metric] / count if count > 0 else 0.0
        metric_values[metric].append(avg_value)

# Access and print metrics from the last report
print("Metrics for the last report (excluding 'O'):")
for metric in metrics:
    value = metric_values[metric][-1]
    print(f"{metric.title()}: {value:.4f}")

# Plot all metrics in one figure
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(
        range(1, len(metric_values[metric]) + 1),
        metric_values[metric],
        marker='o',
        label=metric.title()
    )

plt.title("Metrics Over Active Learning Rounds (Excluding 'O')")
plt.xlabel("Active Learning Round")
plt.ylabel("Score")
plt.xticks(np.arange(1, len(reports) + 1, 1))
plt.yticks(np.arange(0.0, 1.01, 0.1))
plt.ylim(0.0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/V2/Together/V2_Excl_O_metrics_over_rounds.png")
# plt.show()



# # === Metric Extraction and Plotting Separate ===
# # Collect all keys from the first report (assumed consistent across all)
# all_targets = reports[0].keys()
# metrics = ["precision", "recall", "f1-score"]
#
# # Loop through each target label (e.g. 'Auflage', 'macro avg', etc.)
# for target in all_targets:
#     metric_values = {metric: [] for metric in metrics}
#
#     for report in reports:
#         if target in report:  # in case some targets are missing in some rounds
#             for metric in metrics:
#                 metric_values[metric].append(report[target][metric])
#         else:
#             for metric in metrics:
#                 metric_values[metric].append(float('nan'))  # fill missing with NaN
#
#     # Plotting
#     plt.figure()
#     for metric in metrics:
#         plt.plot(metric_values[metric], marker='o', label=metric.title())
#
#     plt.title(f"Metrics for '{target}' Over Active Learning Rounds")
#     plt.xlabel("Active Learning Round")
#     plt.ylabel("Score")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#
#     # Save each plot with a sanitized filename
#     sanitized_name = target.replace(" ", "_").replace(":", "_")
#     plt.savefig(f"plots/V5/Separate/metrics_{sanitized_name}.png")
#     plt.close()

print("Done")