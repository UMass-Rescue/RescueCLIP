import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

# Load the data from the CSV file
file_path = "./scripts/clip_encoding_performance/clip_embed_images_experiment.csv"
df = pd.read_csv(file_path)

# Group by function and batch_size to calculate the mean time
grouped_df = df.groupby(["function", "batch_size"], as_index=False)["time_taken_secs"].mean()

# Define custom color palette for the two function categories
palette = {
    "load_all_the_images_and_then_encode_them_together": "skyblue",
    "load_each_image_and_encode_immediately": "orange",
}

# Calculate slopes for each function group
slopes = {}
for function, group in grouped_df.groupby("function"):
    x = group["batch_size"]
    y = group["time_taken_secs"]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    slopes[function] = slope

# Print the slopes
for function, slope in slopes.items():
    print(f"Slope for '{function}': {slope:.4f} seconds per batch size unit")

# Create the grouped dot plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=grouped_df,  # type: ignore
    x="batch_size",
    y="time_taken_secs",
    hue="function",
    style="function",
    palette=palette,
    s=100,  # Size of the dots
)

# Customize the plot
plt.title("Grouped Dot Plot: Batch Size vs. Time Taken (Averaged)", fontsize=16)
plt.xlabel("Batch Size", fontsize=14)
plt.ylabel("Average Time Taken (seconds)", fontsize=14)
# plt.xscale("log")  # Use log scale for batch size if necessary
plt.xticks(sorted(df["batch_size"].unique()), fontsize=12)
plt.legend(title="Function", fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
