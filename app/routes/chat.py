# chat.py
"""
Endpoints de chat da API.

Endpoints:
    - POST /chat - Envia pergunta ao chatbot
    - GET /chat/suggestions - Retorna sugestões de perguntas
    - GET /chat/history - Retorna histórico de conversas
    - DELETE /chat/history - Limpa histórico
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

# Serviço de chat (será injetado via dependency)
chat_service: Optional[ChatService] = None


def get_service() -> ChatService:
    """Retorna o serviço de chat."""
    if chat_service is None:
        raise HTTPException(status_code=500, detail="Chat service não configurado")
    return chat_service


class ChatRequest(BaseModel):
    """Requisição de chat."""
    pergunta: str = Field(..., min_length=1, max_length=500, description="Pergunta do usuário")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pergunta": "Quantos alunos estão no perfil Crítico?"
            }
        }


class ChatResponseSchema(BaseModel):
    """Resposta do chat."""
    resposta: str = Field(..., description="Resposta do chatbot")
    dados: Optional[List[Dict[str, Any]]] = Field(None, description="Dados retornados (se houver)")
    sql_utilizado: Optional[str] = Field(None, description="SQL gerado (para debug)")
    tipo: str = Field(..., description="Tipo de resposta: query, conversation, error")
    
    class Config:
        json_schema_extra = {
            "example": {
                "resposta": "Existem 110 alunos no perfil Crítico.",
                "dados": [{"perfil": "Crítico", "total": 110}],
                "sql_utilizado": "SELECT COUNT(*) FROM alunos WHERE perfil = 'Crítico'",
                "tipo": "query"
            }
        }


class SuggestionsResponse(BaseModel):
    """Sugestões de perguntas."""
    sugestoes: List[str]


class HistoryItem(BaseModel):
    """Item do histórico."""
    timestamp: str
    pergunta: str
    tipo: str
    sql_utilizado: Optional[str]
    registros_retornados: int
    duracao_segundos: float


class HistoryResponse(BaseModel):
    """Histórico de conversas."""
    total: int
    historico: List[HistoryItem]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("", response_model=ChatResponseSchema)
async def chat(request: ChatRequest):
    """
    Processa uma pergunta do usuário.
    
    O chatbot usa Text-to-SQL para responder perguntas sobre os alunos.
    
    Args:
        request: Pergunta do usuário
    
    Returns:
        Resposta do chatbot com dados (se aplicável)
    
    Examples:
        - "Quantos alunos temos em cada perfil?"
        - "Quais alunos da turma A estão no perfil Crítico?"
        - "Qual a média de engajamento dos alunos Destaque?"
    """
    logger.info(f"[CHAT API] POST /chat - pergunta: {request.pergunta[:50]}...")
    
    service = get_service()
    
    try:
        response = service.chat(request.pergunta)
        
        return ChatResponseSchema(
            resposta=response.message,
            dados=response.data,
            sql_utilizado=response.query_used,
            tipo=response.intent
        )
        
    except Exception as e:
        logger.error(f"[CHAT API] Erro: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions():
    """
    Retorna sugestões de perguntas para o chatbot.
    
    Returns:
        Lista de perguntas sugeridas
    """
    logger.info("[CHAT API] GET /chat/suggestions")
    
    service = get_service()
    
    return SuggestionsResponse(sugestoes=service.get_suggestions())


@router.get("/history", response_model=HistoryResponse)
async def get_history(limit: int = 10):
    """
    Retorna histórico de conversas.
    
    Args:
        limit: Número máximo de itens (padrão: 10)
    
    Returns:
        Histórico de conversas recentes
    """
    logger.info(f"[CHAT API] GET /chat/history - limit={limit}")
    
    service = get_service()
    
    history = service.get_history(limit)
    
    items = [
        HistoryItem(
            timestamp=item.get('timestamp', ''),
            pergunta=item.get('question', ''),
            tipo=item.get('intent', 'unknown'),
            sql_utilizado=item.get('query_used'),
            registros_retornados=item.get('row_count', 0),
            duracao_segundos=item.get('duration_seconds', 0)
        )
        for item in history
    ]
    
    return HistoryResponse(total=len(items), historico=items)


@router.delete("/history")
async def clear_history():
    """
    Limpa o histórico de conversas.
    
    Returns:
        Confirmação de limpeza
    """
    logger.info("[CHAT API] DELETE /chat/history")
    
    service = get_service()
    service.clear_history()
    
    return {"message": "Histórico limpo com sucesso"}