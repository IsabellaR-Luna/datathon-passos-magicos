# health.py
"""
Endpoint de health check da API.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    """Resposta do health check."""
    status: str
    timestamp: str
    version: str


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Verifica se a API está funcionando.
    
    Returns:
        Status da API
    """
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )