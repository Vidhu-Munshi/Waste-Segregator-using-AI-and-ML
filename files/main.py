from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from database.db import init_db
from routes import detect, ocr, history, report, webcam

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title="WasteVision AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(ocr.router)
app.include_router(history.router)
app.include_router(report.router)
app.include_router(webcam.router)

@app.get("/")
async def root():
    return {"status": "WasteVision AI running"}
