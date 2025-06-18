import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# 1. Plot histogram
def plot_duration_histogram(voltage, bins=20):
    LOG_DIR = "logs"
    patterns = [
        f"pulse 1 - {voltage:+.2f}V.csv",
        f"pulse 1 - {-voltage:+.2f}V.csv",
        f"pulse 0 - {voltage:+.2f}V.csv",
        f"pulse 0 - {-voltage:+.2f}V.csv",
    ]

    all_data = []
    for fname in patterns:
        path = os.path.join(LOG_DIR, fname)
        df = pd.read_csv(path)
        last100 = df.iloc[-100:, 1].values
        all_data.append(last100)

    if not all_data:
        print(f"No data found for voltage {voltage:+.2f}V")
        return

    data = np.concatenate(all_data)
    counts, edges = np.histogram(data, bins=bins)

    plt.figure(figsize=(8, 5))
    plt.bar(edges[:-1], counts, width=np.diff(edges), edgecolor='black', align='edge')
    plt.title(f"Last-100 Durations @ {voltage:+.2f} V")
    plt.xlabel("Duration (steps)")
    plt.ylabel("Total Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    mean   = data.mean()
    std    = data.std()
    mx     = data.max()
    c = (data > 300).sum()
    total = data.size
    per = c/total * 100

    print(f"Voltage {voltage:+.2f}V → Mean: {mean:.2f}, Std: {std:.2f}, Max: {mx:.0f}, episodes > 300 steps: {per}%")

    
# 2. Export all csv files to png images
csv_files = glob.glob("logs/*.csv")

total_times = {}

output_folder = "Images"
os.makedirs(output_folder, exist_ok=True)

for file in csv_files:
    df = pd.read_csv(file)
    episodes = df.iloc[:, 0]      # Column 1 = Episode no.
    durations = df.iloc[:, 1]     # Column 2 = Duration (No. of steps)
    total_time = df.iloc[-1, 2]   # Column 3 = Time elapsed in sec
    
    config_name = os.path.basename(file).replace(".csv", "")
    total_time = total_time / 60
    total_times[config_name] = total_time 
    
    rolling_avg = durations.rolling(window=100).mean()
    rolling_avg[:99] = 0
    
    plt.figure(figsize=(10, 7))
    plt.plot(episodes, durations, label='Episode Duration')
    plt.plot(episodes, rolling_avg, color='orange', linewidth=2, label='100-Episode Average')
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.title(f"{os.path.basename(file)}")
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.splitext(os.path.basename(file))[0] + ".png"
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

    # 3. Print runtime for every training 
    time_df = pd.DataFrame.from_dict(total_times, orient='index', columns=["Total Time (min)"])
    time_df = time_df.sort_values(by="Total Time (min)", ascending=False)

print(time_df)

voltages = [0.90, 0.45, 0.30]
for v in voltages:
    plot_duration_histogram(v)
