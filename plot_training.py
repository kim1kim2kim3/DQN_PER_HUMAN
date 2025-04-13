# plot_training.py
import csv
import matplotlib.pyplot as plt

def load_log(filename):
    episodes = []
    rewards = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
    return episodes, rewards

# 두 로그 파일을 비교 (예: training_log_PER.csv vs training_log_uniform.csv)
ep_per, rewards_per = load_log("training_log_PER.csv")
ep_uniform, rewards_uniform = load_log("training_log_uniform.csv")

plt.figure(figsize=(10, 6))
plt.plot(ep_per, rewards_per, label="PER")
plt.plot(ep_uniform, rewards_uniform, label="Uniform Replay")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward Comparison")
plt.legend()
plt.grid(True)
plt.show()
