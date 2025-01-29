import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

# Accept the file_path as a command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file containing the data.")
args = parser.parse_args()

# Load the data from the CSV file
file_path = args.file_path
file_path = "./scripts/clip_encoding_performance/clip_embed_images_experiment.csv"
df = pd.read_csv(file_path)

# Group by function and batch_size to calculate the mean time
grouped_df = df.groupby(["function", "batch_size"], as_index=False)["time_taken_secs"].mean()

# Define custom color palette for the two function categories
palette = {
    "load_all_the_images_and_then_encode_them_together": "skyblue",
    "load_each_image_and_encode_immediately": "orange",
}

# Calculate slopes and intercepts for each function group
slopes = {}
intercepts = {}

plt.figure(figsize=(12, 8))

# Plot scatter points
sns.scatterplot(
    data=grouped_df,  # type: ignore
    x="batch_size",
    y="time_taken_secs",
    hue="function",
    style="function",
    palette=palette,
    s=100,
)

# Fit and plot regression lines
for function, group in grouped_df.groupby("function"):
    x = group["batch_size"]
    y = group["time_taken_secs"]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Store slope and intercept for debugging
    slopes[function] = slope
    intercepts[function] = intercept

    # Plot the regression line
    x_range = sorted(x)
    y_fit = [slope * xi + intercept for xi in x_range]
    plt.plot(x_range, y_fit, label=f"{function} (Slope: {slope:.4f})", linestyle="--", color=palette[function])  # type: ignore

# Print the slopes
for function, slope in slopes.items():
    print(f"Slope for '{function}': {slope:.4f} seconds per batch size unit")

# Customize the plot
plt.title("Plot: Batch Size vs. Time Taken (Averaged over Trials)", fontsize=16)
plt.xlabel("Batch Size", fontsize=14)
plt.ylabel("Average Time Taken (seconds)", fontsize=14)

# Adjust x-axis label rotation and spacing
plt.xticks(ticks=sorted(df["batch_size"].unique()), rotation=45, fontsize=12, ha="right")

plt.legend(title="Function", fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
