import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# 1. Plot histogram
def plot_duration_histogram(pulse, voltage, bins=20):
    filename = f"pulse {pulse} - {voltage:+.2f}V.csv"
    filepath = os.path.join("logs", filename)

    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    durations = df.iloc[:, 1] 

    last100 = durations[-100:] 

    plt.figure(figsize=(8, 5))
    plt.hist(last100, bins=bins, color='skyblue', edgecolor='black', density=True)
    plt.title(f"Histogram of Episode Durations\n(pulse={pulse}, V={voltage:+.2f}V)")
    plt.xlabel("Episode Duration (Reward)")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Mean:", last100.mean())
    print("Std Dev:", last100.std())
    print("Max:", last100.max())
    print("Last 100 Episodes > 300 steps:", (last100 > 300).sum())


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
    
    plt.figure(figsize=(8, 5))
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
    time_df = pd.DataFrame.from_dict(total_times, orient='index', columns=["Total Time (s)"])
    time_df = time_df.sort_values(by="Total Time (s)", ascending=False)

print(time_df)
plot_duration_histogram(pulse=1, voltage=0.9)
