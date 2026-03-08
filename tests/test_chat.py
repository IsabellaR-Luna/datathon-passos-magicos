# test_chat.py
"""
Testes para o módulo de chat.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from app.services.chat.core import (
    QueryExecutor,
    SQLResult,
    QueryResult,
    ChatResponse,
    ChatError,
    SQLGenerationError,
    QueryExecutionError
)


class TestQueryExecutor:
    """Testes para a classe QueryExecutor."""
    
    def test_init(self, temp_db):
        """Testa inicialização do executor."""
        executor = QueryExecutor(db_path=temp_db)
        
        assert executor.db_path == temp_db
    
    def test_execute_select(self, temp_db):
        """Testa execução de SELECT."""
        executor = QueryExecutor(db_path=temp_db)
        
        result = executor.execute("SELECT COUNT(*) as total FROM alunos")
        
        assert isinstance(result, QueryResult)
        assert result.success == True
        assert result.row_count == 1
        assert result.data[0]['total'] == 5
    
    def test_execute_select_with_where(self, temp_db):
        """Testa SELECT com WHERE."""
        executor = QueryExecutor(db_path=temp_db)
        
        result = executor.execute("SELECT * FROM alunos WHERE perfil = 'Destaque'")
        
        assert result.success == True
        assert result.row_count == 2
    
    def test_execute_select_with_group_by(self, temp_db):
        """Testa SELECT com GROUP BY."""
        executor = QueryExecutor(db_path=temp_db)
        
        result = executor.execute(
            "SELECT perfil, COUNT(*) as total FROM alunos GROUP BY perfil"
        )
        
        assert result.success == True
        assert result.row_count > 0
    
    def test_execute_returns_columns(self, temp_db):
        """Testa se execute retorna colunas."""
        executor = QueryExecutor(db_path=temp_db)
        
        result = executor.execute("SELECT ra, nome FROM alunos LIMIT 1")
        
        assert 'ra' in result.columns
        assert 'nome' in result.columns
    
    def test_execute_invalid_sql(self, temp_db):
        """Testa SQL inválido."""
        executor = QueryExecutor(db_path=temp_db)
        
        result = executor.execute("SELECT * FROM tabela_inexistente")
        
        assert result.success == False
        assert result.error is not None
    
    def test_execute_empty_result(self, temp_db):
        """Testa resultado vazio."""
        executor = QueryExecutor(db_path=temp_db)
        
        result = executor.execute("SELECT * FROM alunos WHERE ra = 'INEXISTENTE'")
        
        assert result.success == True
        assert result.row_count == 0
        assert result.data == []


class TestSQLResult:
    """Testes para a dataclass SQLResult."""
    
    def test_create_success(self):
        """Testa criação com sucesso."""
        result = SQLResult(
            success=True,
            query="SELECT * FROM alunos",
            intent="query"
        )
        
        assert result.success == True
        assert result.query == "SELECT * FROM alunos"
        assert result.intent == "query"
    
    def test_create_failure(self):
        """Testa criação com falha."""
        result = SQLResult(
            success=False,
            error="Erro de teste"
        )
        
        assert result.success == False
        assert result.error == "Erro de teste"
    
    def test_default_values(self):
        """Testa valores padrão."""
        result = SQLResult(success=True)
        
        assert result.query is None
        assert result.intent == "unknown"
        assert result.error is None


class TestQueryResult:
    """Testes para a dataclass QueryResult."""
    
    def test_create_success(self):
        """Testa criação com sucesso."""
        result = QueryResult(
            success=True,
            data=[{'id': 1}, {'id': 2}],
            columns=['id'],
            row_count=2
        )
        
        assert result.success == True
        assert len(result.data) == 2
        assert result.row_count == 2
    
    def test_default_values(self):
        """Testa valores padrão."""
        result = QueryResult(success=True)
        
        assert result.data == []
        assert result.columns == []
        assert result.row_count == 0


class TestChatResponse:
    """Testes para a dataclass ChatResponse."""
    
    def test_create(self):
        """Testa criação."""
        response = ChatResponse(
            message="Existem 5 alunos.",
            intent="query"
        )
        
        assert response.message == "Existem 5 alunos."
        assert response.intent == "query"
    
    def test_with_data(self):
        """Testa com dados."""
        response = ChatResponse(
            message="Resultado",
            data=[{'id': 1}],
            query_used="SELECT * FROM alunos",
            intent="query"
        )
        
        assert response.data is not None
        assert response.query_used is not None


class TestSQLGenerator:
    """Testes para a classe SQLGenerator (com mocks)."""
    
    @patch('app.services.chat.core.genai')
    def test_init(self, mock_genai, mock_api_key):
        """Testa inicialização."""
        from app.services.chat.core import SQLGenerator
        
        generator = SQLGenerator(api_key=mock_api_key)
        
        mock_genai.configure.assert_called_once_with(api_key=mock_api_key)
    
    @patch('app.services.chat.core.genai')
    def test_generate_returns_sql_result(self, mock_genai, mock_api_key):
        """Testa se generate retorna SQLResult."""
        from app.services.chat.core import SQLGenerator
        
        # Mock da resposta do modelo
        mock_response = Mock()
        mock_response.text = "SELECT COUNT(*) FROM alunos"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        generator = SQLGenerator(api_key=mock_api_key)
        result = generator.generate("Quantos alunos temos?")
        
        assert isinstance(result, SQLResult)
        assert result.success == True
        assert "SELECT" in result.query
    
    @patch('app.services.chat.core.genai')
    def test_generate_conversation(self, mock_genai, mock_api_key):
        """Testa pergunta que não precisa de SQL."""
        from app.services.chat.core import SQLGenerator
        
        mock_response = Mock()
        mock_response.text = "NAO_SQL"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        generator = SQLGenerator(api_key=mock_api_key)
        result = generator.generate("Olá, como vai?")
        
        assert result.success == True
        assert result.query is None
        assert result.intent == "conversation"
    
    @patch('app.services.chat.core.genai')
    def test_clean_sql_removes_markdown(self, mock_genai, mock_api_key):
        """Testa remoção de markdown."""
        from app.services.chat.core import SQLGenerator
        
        mock_response = Mock()
        mock_response.text = "```sql\nSELECT * FROM alunos\n```"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        generator = SQLGenerator(api_key=mock_api_key)
        result = generator.generate("Liste alunos")
        
        assert "```" not in result.query
        assert result.query == "SELECT * FROM alunos"
    
    @patch('app.services.chat.core.genai')
    def test_validate_sql_blocks_dangerous(self, mock_genai, mock_api_key):
        """Testa bloqueio de SQL perigoso."""
        from app.services.chat.core import SQLGenerator
        
        mock_response = Mock()
        mock_response.text = "DROP TABLE alunos"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        generator = SQLGenerator(api_key=mock_api_key)
        result = generator.generate("Delete tudo")
        
        assert result.success == False


class TestResponseFormatter:
    """Testes para a classe ResponseFormatter (com mocks)."""
    
    @patch('app.services.chat.core.genai')
    def test_format_simple_count(self, mock_genai, mock_api_key):
        """Testa formatação de contagem simples."""
        from app.services.chat.core import ResponseFormatter
        
        formatter = ResponseFormatter(api_key=mock_api_key)
        
        query_result = QueryResult(
            success=True,
            data=[{'total': 5}],
            columns=['total'],
            row_count=1
        )
        
        response = formatter.format("Quantos alunos?", query_result)
        
        assert "5" in response
    
    @patch('app.services.chat.core.genai')
    def test_format_empty_result(self, mock_genai, mock_api_key):
        """Testa formatação de resultado vazio."""
        from app.services.chat.core import ResponseFormatter
        
        formatter = ResponseFormatter(api_key=mock_api_key)
        
        query_result = QueryResult(
            success=True,
            data=[],
            columns=[],
            row_count=0
        )
        
        response = formatter.format("Busca vazia", query_result)
        
        assert "nenhum" in response.lower()
    
    @patch('app.services.chat.core.genai')
    def test_format_error(self, mock_genai, mock_api_key):
        """Testa formatação de erro."""
        from app.services.chat.core import ResponseFormatter
        
        formatter = ResponseFormatter(api_key=mock_api_key)
        
        response = formatter.format_error("Erro de teste")
        
        assert "Erro" in response or "erro" in response
