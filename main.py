from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="TruckGuard AI",
    description=(
        "Production AI/ML platform for federal trucking Hours-of-Service "
        "risk scoring, rule checks, ML prediction, and plain-English "
        "compliance support."
    ),
    version="0.1.0",
)

app.include_router(router)
