from __future__ import annotations

from fastapi import FastAPI

from api.routes import router

app = FastAPI(title="ShelfSense AI API", version="0.1.0")
app.include_router(router)
