# __init__.py
"""
Pacote de chat com Text-to-SQL.

Módulos:
    - core: Classes auxiliares (SQLGenerator, QueryExecutor, ResponseFormatter)
    - chat_service: Serviço principal (orquestrador)
"""

from app.services.chat.core import (
    # Classes
    SQLGenerator,
    QueryExecutor,
    ResponseFormatter,
    # Dataclasses
    SQLResult,
    QueryResult,
    ChatResponse,
    # Exceções
    ChatError,
    SQLGenerationError,
    QueryExecutionError
)

__all__ = [
    # Classes
    'SQLGenerator',
    'QueryExecutor',
    'ResponseFormatter',
    # Dataclasses
    'SQLResult',
    'QueryResult',
    'ChatResponse',
    # Exceções
    'ChatError',
    'SQLGenerationError',
    'QueryExecutionError'
]