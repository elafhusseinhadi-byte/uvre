# =====================================================
# main.py — AI UAV Server (Single City, Prediction, Avoidance)
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
import time, os, math

# -------------------------
# SQLAlchemy Setup
# -------------------------
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    MetaData, Table
)
from sqlalchemy.orm import sessionmaker

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
    Column("system_case", String)
)

metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# =====================================================
# FASTAPI App
# =====================================================
app = FastAPI(title="UAV AI Server – Single City")

# Render PORT
PORT = int(os.environ.get("PORT", 10000))

# =====================================================
# UAV Model
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
    system_case: str

# =====================================================
# Helper Functions (Prediction + Avoidance)
# =====================================================

def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def predict_next(uav, dt=1.0):
    return (
        uav.x + uav.vx * dt,
        uav.y + uav.vy * dt
    )

def avoidance_B(uav):
    uav.x += 0.005
    uav.y -= 0.005

def avoidance_C(uav):
    uav.altitude += 2

def apply_avoidance(a, b):
    avoidance_B(a)
    avoidance_C(b)

# =====================================================
# PUT /uav
# =====================================================
@app.put("/uav")
def put_uav(data: UAV):
    session = SessionLocal()
    start = time.time()

    try:
        existing = session.query(uav_table).filter_by(
            uav_id=data.uav_id
        ).first()

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

        latency = (time.time() - start) * 1000
        return {"status": "ok", "latency_ms": round(latency,3)}

    finally:
        session.close()

# =====================================================
# GET /uavs
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
# POST /process (AI + Avoidance)
# =====================================================
@app.post("/process")
def process_uavs():
    session = SessionLocal()
    start = time.time()

    try:
        uavs = session.query(uav_table).all()
        n = len(uavs)

        collisions = 0
        warnings = 0

        for i in range(n):
            for j in range(i+1, n):
                a = uavs[i]
                b = uavs[j]

                # current distance
                d_now = dist(a, b)

                # predicted next-second distance
                ax2, ay2 = predict_next(a)
                bx2, by2 = predict_next(b)
                d_future = math.sqrt((ax2 - bx2)**2 + (ay2 - by2)**2)

                # collision now
                if d_now < 0.01:
                    collisions += 1

                # danger in the future
                if d_future < 0.02:
                    warnings += 1
                    apply_avoidance(a, b)

        # update database after avoidance
        for u in uavs:
            stmt = (
                uav_table.update()
                .where(uav_table.c.uav_id == u.uav_id)
                .values(
                    x=u.x,
                    y=u.y,
                    altitude=u.altitude
                )
            )
            session.execute(stmt)

        session.commit()

        latency = (time.time() - start)*1000

        return {
            "processed": n,
            "collisions": collisions,
            "future_warnings": warnings,
            "latency_ms": round(latency,3)
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
# Run (Render)
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
