# ============================================================
# FULL UAV ANALYSIS – BEFORE/AFTER + 3 LATENCY CURVES
# Server: https://uvre.onrender.com
# ============================================================

import requests
import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt

BASE = "https://uvre.onrender.com"   # سيرفرك

# ============================================================
# Helper Functions
# ============================================================

def dist(a, b):
    return math.sqrt((a["x"] - b["x"])**2 + (a["y"] - b["y"])**2)

def classify(uavs, thr_actual=0.01, thr_pred=0.03):
    """
    Red     = Actual Collision (قريب جداً)
    Yellow  = Predicted Collision (خطر متوقع)
    Blue    = Normal
    """
    normal, predicted, actual = [], [], []

    for i in range(len(uavs)):
        ui = uavs[i]
        dmin = 999

        for j in range(len(uavs)):
            if i == j:
                continue
            uj = uavs[j]
            d = dist(ui, uj)
            if d < dmin:
                dmin = d

        if dmin < thr_actual:
            actual.append(ui)
        elif dmin < thr_pred:
            predicted.append(ui)
        else:
            normal.append(ui)

    return normal, predicted, actual

def plot_clean(ax, uavs, title):
    normal, predicted, actual = classify(uavs)

    # Normal
    ax.scatter(
        [u["x"] for u in normal],
        [u["y"] for u in normal],
        c="blue", s=25, label="Normal"
    )

    # Predicted (AI)
    ax.scatter(
        [u["x"] for u in predicted],
        [u["y"] for u in predicted],
        c="yellow", edgecolors="black", s=45, label="Predicted Collision"
    )

    # Actual collisions
    ax.scatter(
        [u["x"] for u in actual],
        [u["y"] for u in actual],
        c="red", s=55, label="Actual Collision"
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)

# ============================================================
# Generate Scenario – 100 UAVs
# ============================================================

def generate_100():
    print("🔹 Generating 100 UAVs on server ...")

    # 10 خطر حقيقي (مجموعة قريبة)
    for i in range(10):
        payload = {
            "uav_id": i,
            "x": 33.30 + random.uniform(-0.01, 0.01),
            "y": 44.40 + random.uniform(-0.01, 0.01),
            "altitude": random.uniform(90, 95),
            "speed": random.uniform(20, 60),
            "heading": random.uniform(0, 360),
            "vx": random.uniform(-0.03, 0.03),
            "vy": random.uniform(-0.03, 0.03),
            "system_case": "normal"
        }
        requests.put(f"{BASE}/uav", json=payload)

    # 90 آمنة منتشرة
    for i in range(10, 100):
        payload = {
            "uav_id": i,
            "x": 33.30 + random.uniform(-0.4, 0.4),
            "y": 44.40 + random.uniform(-0.4, 0.4),
            "altitude": random.uniform(95, 120),
            "speed": random.uniform(20, 60),
            "heading": random.uniform(0, 360),
            "vx": random.uniform(-0.03, 0.03),
            "vy": random.uniform(-0.03, 0.03),
            "system_case": "normal"
        }
        requests.put(f"{BASE}/uav", json=payload)

# ============================================================
# Latency Measurements (PUT / GET / POST)
# ============================================================

def measure_latency_put(n=50):
    times = []
    for _ in range(n):
        payload = {
            "uav_id": 9999,
            "x": 33.3,
            "y": 44.4,
            "altitude": 100,
            "speed": 0,
            "heading": 0,
            "vx": 0,
            "vy": 0,
            "system_case": "lat_test"
        }
        t0 = time.time()
        requests.put(f"{BASE}/uav", json=payload)
        times.append(time.time() - t0)
    return np.array(times)

def measure_latency_get(n=50):
    times = []
    for _ in range(n):
        t0 = time.time()
        requests.get(f"{BASE}/uavs")
        times.append(time.time() - t0)
    return np.array(times)

def measure_latency_post(n=50):
    times = []
    for _ in range(n):
        t0 = time.time()
        requests.post(f"{BASE}/process")
        times.append(time.time() - t0)
    return np.array(times)

def plot_latency(ax, values, title):
    mean = values.mean()
    std  = values.std()

    ax.plot(values, color="green", marker="o")
    ax.axhline(mean, color="red", linestyle="--", label=f"Mean = {mean:.3f}s")
    ax.fill_between(
        range(len(values)),
        mean - std,
        mean + std,
        color="green", alpha=0.15
    )
    ax.set_title(title)
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Seconds")
    ax.grid(True)
    ax.legend()

# ============================================================
# RUN FULL WORKFLOW
# ============================================================

def run_all():
    # 1) توليد السيناريو
    generate_100()

    # 2) BEFORE
    before = requests.get(f"{BASE}/uavs").json()["uavs"]

    # 3) AI + Avoidance
    print("🔹 Running AI /process ...")
    info = requests.post(f"{BASE}/process").json()
    print("Process info:", info)

    # 4) AFTER
    after = requests.get(f"{BASE}/uavs").json()["uavs"]

    # ------------------ BEFORE & AFTER جنب بعض ------------------
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    plot_clean(axs[0], before, "🟦 BEFORE AI Avoidance (100 UAVs)")
    plot_clean(axs[1], after,  "🟩 AFTER AI Avoidance (100 UAVs)")

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left")
    plt.tight_layout()
    plt.show()

    # ------------------ 3 كيرفات Latency جنب بعض ----------------
    put_t  = measure_latency_put()
    get_t  = measure_latency_get()
    post_t = measure_latency_post()

    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))
    plot_latency(axs2[0], put_t,  "PUT Latency (/uav)")
    plot_latency(axs2[1], get_t,  "GET Latency (/uavs)")
    plot_latency(axs2[2], post_t, "POST Latency (/process)")
    plt.tight_layout()
    plt.show()

# ============================================================
# RUN
# ============================================================

run_all()
