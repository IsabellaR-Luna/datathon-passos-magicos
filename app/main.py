# main.py
"""
Entry point da API FastAPI.

Uso:
    uvicorn app.main:app --reload --port 8000
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes import health_router, cluster_router, chat_router
from app.routes import cluster as cluster_module
from app.routes import chat as chat_module
from app.services.chat_service import ChatService, ChatServiceConfig
from app.services.chat.core import QueryExecutor
import uvicorn
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DB_PATH = os.getenv("DB_PATH", "data/passos_magicos.db")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
STATIC_DIR = Path(__file__).parent / "static"



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa serviços no startup e limpa no shutdown."""
    
    logger.info("=" * 60)
    logger.info("[API] INICIANDO SERVIÇOS")
    logger.info("=" * 60)
    
    # Inicializa QueryExecutor para rotas de cluster
    cluster_module.query_executor = QueryExecutor(db_path=DB_PATH)
    logger.info(f"[API] QueryExecutor inicializado: {DB_PATH}")
    
    # Inicializa ChatService
    if GOOGLE_API_KEY:
        config = ChatServiceConfig(
            google_api_key=GOOGLE_API_KEY,
            db_path=DB_PATH
        )
        chat_module.chat_service = ChatService(config)
        logger.info("[API] ChatService inicializado")
    else:
        logger.warning("[API] GOOGLE_API_KEY não configurada - chat desabilitado")
    
    logger.info("=" * 60)
    logger.info("[API] SERVIÇOS PRONTOS")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("[API] Encerrando serviços...")


app = FastAPI(
    title="Passos Mágicos API",
    description="API para análise de alunos da Associação Passos Mágicos",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas da API
app.include_router(health_router)
app.include_router(cluster_router)
app.include_router(chat_router)


@app.get("/")
async def serve_frontend():
    """Serve o frontend React."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api")
async def api_info():
    """Informações da API."""
    return {
        "message": "Passos Mágicos API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)