from fastapi import FastAPI
from pydantic import BaseModel
import os, time, math, random

from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import sessionmaker, declarative_base

import numpy as np
from sklearn.linear_model import LogisticRegression

# =====================================================
# DATABASE Setup (ORM)
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
# AI Risk Model — Logistic Regression
# =====================================================

def generate_training_data(n=5000):
    X = []
    y = []

    for _ in range(n):
        d = random.uniform(0.0, 0.10)
        rel_v = random.uniform(0.0, 50.0)
        alt_diff = random.uniform(0.0, 20.0)

        risk = 1 if (d < 0.02 and alt_diff < 5) or (d < 0.03 and rel_v > 20) else 0

        X.append([d, rel_v, alt_diff])
        y.append(risk)

    return np.array(X), np.array(y)

X_train, y_train = generate_training_data()
risk_model = LogisticRegression()
risk_model.fit(X_train, y_train)

# =====================================================
# Helper Functions
# =====================================================

def dist(u1, u2):
    return math.sqrt((u1.x - u2.x)**2 + (u1.y - u2.y)**2)


# *** THE FIX: REAL AVOIDANCE (قوي وواضح بصريًا) ***
def apply_avoidance(u1, u2):

    # الطائرة الأولى تهرب يمين/أعلى
    u1.x += random.uniform(0.05, 0.12)
    u1.y += random.uniform(0.05, 0.12)
    u1.altitude += random.uniform(5, 15)
    u1.vx += random.uniform(0.02, 0.05)
    u1.vy += random.uniform(0.02, 0.05)
    u1.system_case = "ai_avoid"

    # الطائرة الثانية تهرب يسار/أسفل
    u2.x -= random.uniform(0.05, 0.12)
    u2.y -= random.uniform(0.05, 0.12)
    u2.altitude -= random.uniform(5, 15)
    u2.vx -= random.uniform(0.02, 0.05)
    u2.vy -= random.uniform(0.02, 0.05)
    u2.system_case = "ai_avoid"


# =====================================================
# FASTAPI APP
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
    system_case: str = "normal"

# =====================================================
# PUT /uav
# =====================================================

@app.put("/uav")
def put_uav(data: UAV):
    session = SessionLocal()
    try:
        existing = session.query(UAVModel).filter_by(uav_id=data.uav_id).first()

        if existing:
            existing.x = data.x
            existing.y = data.y
            existing.altitude = data.altitude
            existing.speed = data.speed
            existing.heading = data.heading
            existing.vx = data.vx
            existing.vy = data.vy
            existing.system_case = data.system_case
        else:
            obj = UAVModel(
                uav_id=data.uav_id,
                x=data.x, y=data.y,
                altitude=data.altitude,
                speed=data.speed,
                heading=data.heading,
                vx=data.vx, vy=data.vy,
                system_case=data.system_case
            )
            session.add(obj)

        session.commit()
        return {"status": "ok"}
    finally:
        session.close()

# =====================================================
# GET /uavs
# =====================================================

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
                    x=u.x,
                    y=u.y,
                    altitude=u.altitude,
                    speed=u.speed,
                    heading=u.heading,
                    vx=u.vx,
                    vy=u.vy,
                    system_case=u.system_case
                ) for u in rows
            ]
        }
    finally:
        session.close()

# =====================================================
# POST /process — AI Prediction + Avoidance
# =====================================================

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
            d_min = 999

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

            d_now  = d_min
            rel_v  = math.sqrt((ui.vx - nearest.vx)**2 + (ui.vy - nearest.vy)**2)
            alt_d  = abs(ui.altitude - nearest.altitude)

            if d_now < 0.01:
                collisions += 1

            P = risk_model.predict_proba([[d_now, rel_v, alt_d]])[0, 1]

            if P > 0.6:
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

# =====================================================
# Health
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# =====================================================
# Run
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
