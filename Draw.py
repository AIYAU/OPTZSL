import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')
# Data extracted from the table
models = ['GPT4o', 'Claude', 'Mistral', 'GPT4o-Mini', 'GPT-3.5', 'Google']
word_lengths = ['Within 10 words', 'Within 20 words', 'Within 30 words', 'Within 40 words']

# Accuracy values from the table
accuracy = {
    'Within 10 words': [52.42, 42.30, 63.49, 51.60, 45.20, 47.06],
    'Within 20 words': [54.80, 59.48, 54.87, 46.02, 51.90, 54.80],
    'Within 30 words': [44.91, 53.90, 41.71, 33.01, 40.15, 47.58],
    'Within 40 words': [49.14, 59.55, 58.22, 40.67, 45.87, 49.96]
}

bar_width = 0.2  # Adjust bar width to avoid overlap
index = np.arange(len(models))

# Create the plot with a larger figure size and better layout
fig, ax = plt.subplots(figsize=(9, 5))

# Define colors for each word length category
colors = ['#A1CAF1', '#FFCCCB', '#C5E3BF', '#FFE5B4']

# Plot bars for each word length category
for i, length in enumerate(word_lengths):
    ax.bar(index + i * bar_width, accuracy[length], bar_width, label=length, color=colors[i])

# Add labels and title with enhanced font sizes
ax.set_xlabel('Large Language Model', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Accuracy Across Models for Different Prompt Word Lengths', fontsize=15, fontweight='bold')

# Set xticks and tick parameters for better visibility
ax.set_xticks(index + 1.5 * bar_width / 2)
ax.set_xticklabels(models, fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# Position the legend to the right and add a border for clarity
ax.legend(title='Prompt Word Length', fontsize=10, title_fontsize=12, loc='upper left',
          bbox_to_anchor=(1, 1), frameon=True, framealpha=0.9)

# Add value labels on top of each bar with improved formatting
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset for better spacing
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# Add value labels for each group of bars
for i, length in enumerate(word_lengths):
    bars = ax.bar(index + i * bar_width, accuracy[length], bar_width, color=colors[i])
    add_value_labels(bars)

# Adjust layout for better spacing
plt.tight_layout()

# Display the improved chart
plt.show()
