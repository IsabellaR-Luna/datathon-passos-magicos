# chat_service.py
"""
Serviço principal de chat com Text-to-SQL.

Este serviço orquestra o fluxo de perguntas do professor:
1. Recebe pergunta
2. Gera SQL (se necessário)
3. Executa query
4. Formata resposta

Uso:
    service = ChatService(config)
    response = service.chat("Quantos alunos críticos temos?")
    print(response.message)
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
from app.services.chat.core import (
    SQLGenerator,
    QueryExecutor,
    ResponseFormatter,
    SQLResult,
    QueryResult,
    ChatResponse,
    ChatError,
    SQLGenerationError,
    QueryExecutionError
)

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURAÇÃO
# ===========================================================================

@dataclass
class ChatServiceConfig:
    """Configuração do serviço de chat."""
    
    # Google API
    google_api_key: str = ""
    model_name: str = "gemini-2.5-flash"
    
    # Database
    db_path: str = "data/passos_magicos.db"
    
    # Limites
    max_results: int = 100
    
    def __post_init__(self):
        # Tenta pegar API key do ambiente se não fornecida
        if not self.google_api_key:
            self.google_api_key = os.getenv("GOOGLE_API_KEY", "")


# ============================================================================
# CLASSE PRINCIPAL: ChatService
# ============================================================================

class ChatService:
    """
    Serviço principal de chat com Text-to-SQL.
    
    Fluxo:
        Pergunta → SQLGenerator → QueryExecutor → ResponseFormatter → Resposta
    """
    
    def __init__(self, config: Optional[ChatServiceConfig] = None):
        """
        Inicializa o serviço de chat.
        
        Args:
            config: Configuração do serviço
        """
        self.config = config or ChatServiceConfig()
        
        if not self.config.google_api_key:
            raise ChatError("GOOGLE_API_KEY não configurada")
        
        # Inicializa componentes
        self.sql_generator = SQLGenerator(
            api_key=self.config.google_api_key,
            model_name=self.config.model_name
        )
        
        self.query_executor = QueryExecutor(
            db_path=self.config.db_path
        )
        
        self.response_formatter = ResponseFormatter(
            api_key=self.config.google_api_key,
            model_name=self.config.model_name
        )
        
        # Histórico de conversas
        self._history: List[Dict[str, Any]] = []
        
        logger.info("=" * 60)
        logger.info("[CHAT SERVICE] SERVIÇO INICIALIZADO")
        logger.info(f"[CHAT SERVICE] Modelo: {self.config.model_name}")
        logger.info(f"[CHAT SERVICE] Banco: {self.config.db_path}")
        logger.info("=" * 60)
    
    def chat(self, question: str) -> ChatResponse:
        """
        Processa uma pergunta do usuário.
        
        Args:
            question: Pergunta em linguagem natural
            
        Returns:
            ChatResponse com resposta formatada
        """
        logger.info("=" * 60)
        logger.info(f"[CHAT SERVICE] NOVA PERGUNTA: {question}")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 1. Gera SQL
            logger.info("[CHAT SERVICE] Etapa 1/3: Gerando SQL...")
            sql_result = self.sql_generator.generate(question)
            
            if not sql_result.success:
                logger.error(f"[CHAT SERVICE] Falha na geração de SQL: {sql_result.error}")
                return ChatResponse(
                    message=self.response_formatter.format_error(sql_result.error),
                    intent="error"
                )
            
            # 2. Se não precisa de SQL (conversa geral)
            if sql_result.intent == "conversation":
                logger.info("[CHAT SERVICE] Pergunta conversacional, sem SQL")
                message = self.response_formatter.format_conversation(question)
                
                response = ChatResponse(
                    message=message,
                    intent="conversation"
                )
                
                self._log_interaction(question, response, None, start_time)
                return response
            
            # 3. Executa query
            logger.info("[CHAT SERVICE] Etapa 2/3: Executando query...")
            query_result = self.query_executor.execute(sql_result.query)
            
            if not query_result.success:
                logger.error(f"[CHAT SERVICE] Falha na execução: {query_result.error}")
                return ChatResponse(
                    message=self.response_formatter.format_error(query_result.error),
                    query_used=sql_result.query,
                    intent="error"
                )
            
            # 4. Formata resposta
            logger.info("[CHAT SERVICE] Etapa 3/3: Formatando resposta...")
            message = self.response_formatter.format(
                question=question,
                query_result=query_result,
                sql_used=sql_result.query
            )
            
            response = ChatResponse(
                message=message,
                data=query_result.data if query_result.row_count <= 20 else None,
                query_used=sql_result.query,
                intent="query"
            )
            
            self._log_interaction(question, response, query_result, start_time)
            
            logger.info("=" * 60)
            logger.info("[CHAT SERVICE] RESPOSTA GERADA COM SUCESSO")
            logger.info("=" * 60)
            
            return response
            
        except Exception as e:
            logger.error(f"[CHAT SERVICE] ERRO NÃO TRATADO: {str(e)}")
            return ChatResponse(
                message=f"Ocorreu um erro inesperado: {str(e)}",
                intent="error"
            )
    
    def get_suggestions(self) -> List[str]:
        """
        Retorna sugestões de perguntas.
        
        Returns:
            Lista de perguntas sugeridas
        """
        return [
            "Quantos alunos temos em cada perfil?",
            "Quais alunos do perfil Crítico precisam de atenção urgente?",
            "Qual a média de engajamento (IEG) por turma?",
            "Liste os alunos que atingiram o Ponto de Virada",
            "Quais alunos têm maior defasagem escolar?",
            "Mostre os 10 alunos com melhor desempenho (IDA)",
            "Quantos alunos foram indicados para bolsa?",
            "Qual a distribuição de alunos por pedra em 2022?",
            "Quais alunos do perfil Atenção têm IEG acima de 8?",
            "Compare as médias de notas entre os perfis"
        ]
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retorna histórico de conversas.
        
        Args:
            limit: Número máximo de interações
            
        Returns:
            Lista de interações recentes
        """
        return self._history[-limit:]
    
    def clear_history(self):
        """Limpa histórico de conversas."""
        self._history = []
        logger.info("[CHAT SERVICE] Histórico limpo")
    
    def _log_interaction(
        self,
        question: str,
        response: ChatResponse,
        query_result: Optional[QueryResult],
        start_time: datetime
    ):
        """Registra interação no histórico."""
        duration = (datetime.now() - start_time).total_seconds()
        
        interaction = {
            "timestamp": start_time.isoformat(),
            "question": question,
            "intent": response.intent,
            "query_used": response.query_used,
            "row_count": query_result.row_count if query_result else 0,
            "duration_seconds": round(duration, 2)
        }
        
        self._history.append(interaction)
        
        logger.info(f"[CHAT SERVICE] Interação registrada: {duration:.2f}s, {interaction['row_count']} rows")
    
    # ========================================================================
    # MÉTODOS AUXILIARES
    # ========================================================================
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo dos perfis (atalho comum).
        
        Returns:
            Dict com contagem por perfil
        """
        result = self.query_executor.execute(
            "SELECT perfil, COUNT(*) as total FROM alunos GROUP BY perfil ORDER BY total DESC"
        )
        
        if result.success:
            return {row['perfil']: row['total'] for row in result.data}
        return {}
    
    def get_student_info(self, ra: str) -> Optional[Dict[str, Any]]:
        """
        Retorna informações de um aluno específico.
        
        Args:
            ra: Registro do aluno
            
        Returns:
            Dict com dados do aluno ou None
        """
        result = self.query_executor.execute(
            f"SELECT * FROM alunos WHERE ra = '{ra}'"
        )
        
        if result.success and result.row_count > 0:
            return result.data[0]
        return None
    
    def get_students_by_profile(self, profile: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retorna alunos de um perfil específico.
        
        Args:
            profile: Nome do perfil
            limit: Limite de resultados
            
        Returns:
            Lista de alunos
        """
        result = self.query_executor.execute(
            f"SELECT ra, nome, iaa, ieg, ida, defasagem FROM alunos WHERE perfil = '{profile}' LIMIT {limit}"
        )
        
        if result.success:
            return result.data
        return []

