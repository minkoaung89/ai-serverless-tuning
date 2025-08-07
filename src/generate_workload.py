# ====================================
# Section 1: Synthetic Data Generation
# ====================================
# Install necessary tools
!pip install numpy pandas scikit-learn xgboost tensorflow keras prophet gym gymnasium boto3 stable-baselines3[extra] "shimmy>=2.0"

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)

# Configuration
num_days = 30
interval_minutes = 5
total_points = int((24 * 60 / interval_minutes) * num_days)  # ~2016 rows

# Generate timestamps
start_time = datetime(2024, 1, 1, 0, 0, 0)
timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(total_points)]

# === Predictable Bursty Invocations ===
def generate_bursty_invocations(n, base=100):
    invocations = []
    for i in range(n):
        # Inject burst every 50 steps (approx every 4.2 hrs)
        if i % 50 in range(5):  # 5-step burst duration
            val = base + 120 + np.random.normal(0, 10)
        else:
            val = base + np.sin(i / 10.0) * 25 + np.random.normal(0, 5)
        invocations.append(max(1, int(val)))
    return invocations

invocations = generate_bursty_invocations(total_points)

# Memory allocations
memory_choices = [128, 256, 512, 1024, 2048, 3008]
memory_MB = [random.choice(memory_choices) for _ in range(total_points)]

# Timeout (in seconds): AWS Lambda typical range
timeout_sec = [random.choice([1, 3, 5, 10, 15, 30]) for _ in range(total_points)]

# Concurrency estimate (inferred from invocation rate)
concurrency = [max(1, int(inv / 10)) for inv in invocations]

# Cold start (10% probability)
cold_start = [1 if random.random() < 0.1 else 0 for _ in range(total_points)]

# Duration (ms): base duration affected by load
duration_ms = [round(random.gauss(100 + inv / 3, 15), 2) for inv in invocations]

# Latency (ms): base + cold start penalty + noise
latency_ms = [
    round(d + (150 if cs else 20) + np.random.normal(0, 10), 2)
    for d, cs in zip(duration_ms, cold_start)
]

# Create DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "invocations": invocations,
    "memory_MB": memory_MB,
    "timeout_sec": timeout_sec,
    "concurrency": concurrency,
    "cold_start": cold_start,
    "latency_ms": latency_ms,
    "duration_ms": duration_ms,
})

# Timestamp feature engineering
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["second"] = df["timestamp"].dt.second
df["dayofweek"] = df["timestamp"].dt.dayofweek

# Lag features
df["invocations_lag1"] = df["invocations"].shift(1)
df["invocations_lag2"] = df["invocations"].shift(2)
df["invocations_avg_5"] = df["invocations"].rolling(window=5).mean()
df["invocations_std_5"] = df["invocations"].rolling(window=5).std()

# Drop initial NaN rows
df = df.dropna().reset_index(drop=True)

# Export
df.to_csv("synthetic_invocations.csv", index=False)
print(df.head())


# =================================================================
# Section 7: Multi-Traffic Evaluation (Bursty, Periodic, Irregular)
# =================================================================

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Load trained A2C model
a2c_model = A2C.load("a2c_serverless_model")

# Enhanced traffic pattern generator with realistic variation
def generate_traffic(pattern, total_points):
    base = 100
    if pattern == "bursty":
        return [int(base + (np.sin(i / 15.0) * 60 + (i % 20) * 10 + np.random.normal(0, 10))) for i in range(total_points)]
    elif pattern == "periodic":
        return [int(base + np.cos(i / 10.0) * 50 + 30 * np.sin(i / 30.0) + np.random.normal(0, 10)) for i in range(total_points)]
    elif pattern == "irregular":
        return [int(base + random.randint(-120, 180)) for i in range(total_points)]
    else:
        raise ValueError("Invalid traffic pattern")

# Evaluation function
def evaluate_with_a2c(forecast_seq, model):
    env = DummyVecEnv([lambda: ServerlessTuningEnv(forecast_seq)])
    obs = env.reset()
    done = False

    latency_list = []
    cost_list = []
    sla_violations = 0
    total_steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        latency = info[0]["latency"]
        cost = info[0]["cost"]
        violated = info[0]["sla_violated"]

        latency_list.append(latency)
        cost_list.append(cost)
        sla_violations += int(violated)
        total_steps += 1

    return pd.DataFrame({
        "Avg_Latency": [np.mean(latency_list)],
        "Avg_Cost": [np.mean(cost_list)],
        "SLA_Violation": [sla_violations / total_steps * 100]
    })

# Generate traffic
total_points = len(forecast_series)
traffic_bursty = generate_traffic("bursty", total_points)
traffic_periodic = generate_traffic("periodic", total_points)
traffic_irregular = generate_traffic("irregular", total_points)

# Evaluate A2C on all patterns
df_bursty = evaluate_with_a2c(traffic_bursty, a2c_model)
df_periodic = evaluate_with_a2c(traffic_periodic, a2c_model)
df_irregular = evaluate_with_a2c(traffic_irregular, a2c_model)

# Summary print
def summarize(df, label):
    print(f"--- {label} ---")
    print(f"Average Latency: {df['Avg_Latency'].mean():.2f} ms")
    print(f"Average Cost: ${df['Avg_Cost'].mean():.6f}")
    print(f"SLA Violation Rate: {df['SLA_Violation'].mean():.2f}%\n")

summarize(df_bursty, "Bursty Traffic")
summarize(df_periodic, "Periodic Traffic")
summarize(df_irregular, "Irregular Traffic")

# Visualization
patterns = ["Bursty", "Periodic", "Irregular"]
latencies = [df_bursty["Avg_Latency"].mean(), df_periodic["Avg_Latency"].mean(), df_irregular["Avg_Latency"].mean()]
costs = [df_bursty["Avg_Cost"].mean(), df_periodic["Avg_Cost"].mean(), df_irregular["Avg_Cost"].mean()]
violations = [df_bursty["SLA_Violation"].mean(), df_periodic["SLA_Violation"].mean(), df_irregular["SLA_Violation"].mean()]

x = np.arange(len(patterns))

# Subplots in one row
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
bar_width = 0.6

# Latency subplot
axs[0].bar(x, latencies, color='skyblue', width=bar_width)
axs[0].set_xticks(x)
axs[0].set_xticklabels(patterns)
axs[0].set_ylabel("Latency (ms)")
axs[0].set_title("Average Latency")
axs[0].grid(axis='y')
for i, val in enumerate(latencies):
    axs[0].text(i, val + 1, f"{val:.2f}", ha='center', fontsize=9)

# Cost subplot
axs[1].bar(x, costs, color='lightgreen', width=bar_width)
axs[1].set_xticks(x)
axs[1].set_xticklabels(patterns)
axs[1].set_ylabel("Cost (USD)")
axs[1].set_title("Average Cost")
axs[1].grid(axis='y')
for i, val in enumerate(costs):
    axs[1].text(i, val + 0.002, f"{val:.5f}", ha='center', fontsize=9)

# SLA Violation subplot
axs[2].bar(x, violations, color='salmon', width=bar_width)
axs[2].set_xticks(x)
axs[2].set_xticklabels(patterns)
axs[2].set_ylabel("Violation Rate (%)")
axs[2].set_title("SLA Violation Rate")
axs[2].grid(axis='y')
for i, val in enumerate(violations):
    axs[2].text(i, val + 1, f"{val:.2f}%", ha='center', fontsize=9)

# Overall layout
fig.suptitle("A2C Evaluation Across Traffic Patterns", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
