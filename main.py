# =====================================================
# main.py — Cloud + AI UAV Server (FINAL – FIXED RISK)
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
import os, time, math, random

from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import sessionmaker, declarative_base

import numpy as np
from sklearn.linear_model import LogisticRegression

# =====================================================
# DATABASE (SQLite)
# =====================================================

DATABASE_URL = "sqlite:///uav_ai.sqlite"

Base = declarative_base()

class UAVModel(Base):
    __tablename__ = "uavs"

    uav_id     = Column(Integer, primary_key=True)
    x          = Column(Float)
    y          = Column(Float)
    altitude   = Column(Float)
    speed      = Column(Float)
    heading    = Column(Float)
    vx         = Column(Float)
    vy         = Column(Float)
    system_case = Column(String)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# =====================================================
# AI MODEL — Logistic Regression
# =====================================================

def generate_training_data(n=5000):
    X = []
    y = []
    for _ in range(n):
        d = random.uniform(0, 0.1)
        rel_v = random.uniform(0, 50)
        alt_diff = random.uniform(0, 20)
        risk = 1 if (d < 0.02 and alt_diff < 5) or (d < 0.03 and rel_v > 20) else 0
        X.append([d, rel_v, alt_diff])
        y.append(risk)
    return np.array(X), np.array(y)

X_train, y_train = generate_training_data()
risk_model = LogisticRegression()
risk_model.fit(X_train, y_train)

# =====================================================
# Utility
# =====================================================

def dist(u1, u2):
    return math.sqrt((u1.x - u2.x)**2 + (u1.y - u2.y)**2)

# قوي — يغيّر بشكل واضح
def apply_avoidance(u1, u2):
    # الأول يهرب يمين/فوق
    u1.x += random.uniform(0.08, 0.15)
    u1.y += random.uniform(0.08, 0.15)
    u1.altitude += random.uniform(8, 18)
    u1.vx += random.uniform(0.03, 0.07)
    u1.vy += random.uniform(0.03, 0.07)
    u1.system_case = "ai_avoid"

    # الثاني يهرب يسار/تحت
    u2.x -= random.uniform(0.08, 0.15)
    u2.y -= random.uniform(0.08, 0.15)
    u2.altitude -= random.uniform(8, 18)
    u2.vx -= random.uniform(0.03, 0.07)
    u2.vy -= random.uniform(0.03, 0.07)
    u2.system_case = "ai_avoid"

# =====================================================
# FastAPI App
# =====================================================

app = FastAPI(title="Cloud+AI UAV Server – FINAL")
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
    system_case: str = "normal"

# =====================================================
# API ENDPOINTS
# =====================================================

# ---------- RESET DATABASE ----------
@app.post("/reset")
def reset_db():
    session = SessionLocal()
    try:
        session.query(UAVModel).delete()
        session.commit()
        return {"status": "database_cleared"}
    finally:
        session.close()

# ---------- PUT /uav ----------
@app.put("/uav")
def put_uav(data: UAV):
    session = SessionLocal()
    try:
        u = session.query(UAVModel).filter_by(uav_id=data.uav_id).first()
        if u:
            u.x = data.x
            u.y = data.y
            u.altitude = data.altitude
            u.speed = data.speed
            u.heading = data.heading
            u.vx = data.vx
            u.vy = data.vy
            u.system_case = data.system_case
        else:
            u = UAVModel(
                uav_id=data.uav_id,
                x=data.x, y=data.y,
                altitude=data.altitude,
                speed=data.speed,
                heading=data.heading,
                vx=data.vx, vy=data.vy,
                system_case=data.system_case
            )
            session.add(u)
        session.commit()
        return {"status": "ok"}
    finally:
        session.close()

# ---------- GET /uavs ----------
@app.get("/uavs")
def get_uavs():
    session = SessionLocal()
    try:
        rows = session.query(UAVModel).all()
        return {
            "count": len(rows),
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
                for u in rows
            ]
        }
    finally:
        session.close()

# ---------- POST /process ----------
@app.post("/process")
def process_uavs():
    session = SessionLocal()
    try:
        uavs = session.query(UAVModel).all()
        n = len(uavs)

        if n == 0:
            return {"processed": 0}

        collisions = 0
        high_risk = 0

        for i in range(n):
            ui = uavs[i]

            nearest = None
            dmin = 999
            for j in range(n):
                if i == j:
                    continue
                uj = uavs[j]
                d = dist(ui, uj)
                if d < dmin:
                    dmin = d
                    nearest = uj

            if nearest is None:
                continue

            d_now = dmin
            rel_v = math.sqrt((ui.vx - nearest.vx)**2 + (ui.vy - nearest.vy)**2)
            alt_d = abs(ui.altitude - nearest.altitude)

            # تصادم فعلي (قريب جداً)
            if d_now < 0.01:
                collisions += 1

            # 🔥 دمج AI + rule-based حتى نضمن high_risk>0
            P = risk_model.predict_proba([[d_now, rel_v, alt_d]])[0, 1]
            rule_risk = 1 if ((d_now < 0.03 and alt_d < 10) or (d_now < 0.05 and rel_v > 10)) else 0

            if P > 0.4 or rule_risk:
                high_risk += 1
                apply_avoidance(ui, nearest)

        session.commit()

        return {
            "processed": n,
            "collisions": collisions,
            "high_risk": high_risk
        }

    finally:
        session.close()

# ---------- HEALTH ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
