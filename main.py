# =====================================================
# main.py — Cloud + AI UAV Server (Single City)
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
import os, time, math, random

from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table
from sqlalchemy.orm import sessionmaker

import numpy as np
from sklearn.linear_model import LogisticRegression

# =====================================================
# DB Setup (SQLite)
# =====================================================
DATABASE_URL = "sqlite:///uav_ai.sqlite"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

metadata = MetaData()

uav_table = Table(
    "uavs", metadata,
    Column("uav_id", Integer, primary_key=True),
    Column("x", Float),
    Column("y", Float),
    Column("altitude", Float),
    Column("speed", Float),
    Column("heading", Float),
    Column("vx", Float),
    Column("vy", Float),
    Column("system_case", String),   # normal / ai_avoid
)

metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# =====================================================
# AI Risk Model (Logistic Regression) – Synthetic Training
# =====================================================

def generate_training_data(n=5000):
    """نولد بيانات صناعية لتعليم نموذج التصادم."""
    X = []
    y = []
    for _ in range(n):
        d = random.uniform(0.0, 0.10)      # distance (deg)
        rel_v = random.uniform(0.0, 50.0)  # relative speed
        alt_diff = random.uniform(0.0, 20.0)

        # قاعدة بسيطة لتوليد الليبل:
        # قريب جداً + فرق ارتفاع قليل -> غالباً خطر
        # قريب ومتقابلين بسرعة عالية -> خطر
        risk = 1 if (d < 0.02 and alt_diff < 5) or (d < 0.03 and rel_v > 20) else 0

        X.append([d, rel_v, alt_diff])
        y.append(risk)

    return np.array(X), np.array(y)

X_train, y_train = generate_training_data()
risk_model = LogisticRegression()
risk_model.fit(X_train, y_train)

# =====================================================
# Helper Functions (Distance + Prediction + Avoidance)
# =====================================================

def dist(u1, u2):
    return math.sqrt((u1.x - u2.x)**2 + (u1.y - u2.y)**2)

def predict_next(u, dt=1.0):
    return (u.x + u.vx*dt, u.y + u.vy*dt)

def avoidance_B(u):
    # side shift بسيط
    u.x += 0.003
    u.y -= 0.003

def avoidance_C(u):
    # تغيير ارتفاع بسيط
    u.altitude += 2.0

def apply_avoidance(u_high, u_other):
    # نطبّق B على الأولى و C على الثانية
    avoidance_B(u_high)
    avoidance_C(u_other)
    u_high.system_case = "ai_avoid"
    u_other.system_case = "ai_avoid"

# =====================================================
# FASTAPI App
# =====================================================

app = FastAPI(title="Cloud+AI UAV Collision Prediction Server")
PORT = int(os.environ.get("PORT", 10000))

# =====================================================
# Pydantic Model
# =====================================================

class UAV(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    speed: float
    heading: float
    vx: float
    vy: float
    system_case: str = "normal"   # default

# =====================================================
# PUT /uav — insert/update single UAV
# =====================================================

@app.put("/uav")
def put_uav(data: UAV):
    session = SessionLocal()
    start = time.time()
    try:
        existing = session.query(uav_table).filter_by(uav_id=data.uav_id).first()

        values = {
            "x": data.x,
            "y": data.y,
            "altitude": data.altitude,
            "speed": data.speed,
            "heading": data.heading,
            "vx": data.vx,
            "vy": data.vy,
            "system_case": data.system_case,
        }

        if existing:
            stmt = (
                uav_table.update()
                .where(uav_table.c.uav_id == data.uav_id)
                .values(**values)
            )
            session.execute(stmt)
        else:
            values["uav_id"] = data.uav_id
            stmt = uav_table.insert().values(**values)
            session.execute(stmt)

        session.commit()
        latency = (time.time() - start)*1000
        return {"status": "ok", "latency_ms": round(latency, 3)}
    finally:
        session.close()

# =====================================================
# GET /uavs — all UAVs
# =====================================================

@app.get("/uavs")
def get_uavs():
    session = SessionLocal()
    try:
        uavs = session.query(uav_table).all()
        return {
            "count": len(uavs),
            "uavs": [
                dict(
                    uav_id=u.uav_id,
                    x=u.x, y=u.y,
                    altitude=u.altitude,
                    speed=u.speed,
                    heading=u.heading,
                    vx=u.vx, vy=u.vy,
                    system_case=u.system_case
                )
                for u in uavs
            ]
        }
    finally:
        session.close()

# =====================================================
# POST /process — AI Prediction + Avoidance
# =====================================================

@app.post("/process")
def process():
    session = SessionLocal()
    start = time.time()
    try:
        uavs = session.query(uav_table).all()
        n = len(uavs)

        if n == 0:
            return {"processed": 0, "collisions": 0, "high_risk": 0, "latency_ms": 0}

        collisions = 0
        high_risk = 0

        # -------- نحسب أقرب جار لكل طائرة + نمرّرها للـ AI ----------
        for i in range(n):
            ui = uavs[i]

            # ابحث عن أقرب جار
            nearest = None
            d_min = 1e9
            for j in range(n):
                if i == j:
                    continue
                uj = uavs[j]
                d = dist(ui, uj)
                if d < d_min:
                    d_min = d
                    nearest = uj

            if nearest is None:
                continue

            # مسافة حالية
            d_now = d_min
            # سرعة نسبية
            rel_v = math.sqrt((ui.vx - nearest.vx)**2 + (ui.vy - nearest.vy)**2)
            # فرق ارتفاع
            alt_diff = abs(ui.altitude - nearest.altitude)

            # Detection rule-based
            if d_now < 0.01:
                collisions += 1

            # AI risk prediction
            X_feat = np.array([[d_now, rel_v, alt_diff]])
            prob = risk_model.predict_proba(X_feat)[0,1]  # احتمال التصادم

            if prob > 0.6:   # threshold
                high_risk += 1
                apply_avoidance(ui, nearest)

        # -------- تحديث قاعدة البيانات بعد الـ Avoidance ----------
        for u in uavs:
            stmt = (
                uav_table.update()
                .where(uav_table.c.uav_id == u.uav_id)
                .values(
                    x=u.x,
                    y=u.y,
                    altitude=u.altitude,
                    system_case=u.system_case
                )
            )
            session.execute(stmt)

        session.commit()
        latency = (time.time() - start)*1000

        return {
            "processed": n,
            "collisions": collisions,
            "high_risk": high_risk,
            "latency_ms": round(latency, 3)
        }
    finally:
        session.close()

# =====================================================
# Health Check
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# =====================================================
# Run locally / Render
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
