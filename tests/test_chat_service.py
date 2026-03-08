# test_chat_service.py
"""
Testes para o serviço de chat completo.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from app.services.chat_service import ChatService, ChatServiceConfig
from app.services.chat.core import ChatResponse, QueryResult


class TestChatServiceConfig:
    """Testes para a configuração do serviço de chat."""
    
    def test_default_values(self):
        """Testa valores padrão."""
        config = ChatServiceConfig()
        
        assert config.model_name == "gemini-1.5-flash"
        assert config.db_path == "data/passos_magicos.db"
        assert config.max_results == 100
    
    def test_custom_values(self):
        """Testa valores customizados."""
        config = ChatServiceConfig(
            google_api_key="test-key",
            db_path="/custom/path.db"
        )
        
        assert config.google_api_key == "test-key"
        assert config.db_path == "/custom/path.db"
    
    def test_reads_env_var(self):
        """Testa leitura de variável de ambiente."""
        os.environ['GOOGLE_API_KEY'] = 'env-test-key'
        
        config = ChatServiceConfig()
        
        assert config.google_api_key == 'env-test-key'
        
        # Cleanup
        del os.environ['GOOGLE_API_KEY']
    
    def test_explicit_key_overrides_env(self):
        """Testa que chave explícita sobrescreve env."""
        os.environ['GOOGLE_API_KEY'] = 'env-key'
        
        config = ChatServiceConfig(google_api_key='explicit-key')
        
        assert config.google_api_key == 'explicit-key'
        
        # Cleanup
        del os.environ['GOOGLE_API_KEY']


class TestChatServiceInit:
    """Testes para inicialização do ChatService."""
    
    def test_init_without_key_raises_error(self):
        """Testa que inicialização sem key levanta erro."""
        # Garante que não há key no ambiente
        if 'GOOGLE_API_KEY' in os.environ:
            del os.environ['GOOGLE_API_KEY']
        
        config = ChatServiceConfig(google_api_key="")
        
        with pytest.raises(Exception):
            ChatService(config)
    
    @patch('app.services.chat_service.SQLGenerator')
    @patch('app.services.chat_service.QueryExecutor')
    @patch('app.services.chat_service.ResponseFormatter')
    def test_init_with_key(self, mock_formatter, mock_executor, mock_generator):
        """Testa inicialização com key válida."""
        config = ChatServiceConfig(
            google_api_key="test-key",
            db_path="test.db"
        )
        
        service = ChatService(config)
        
        assert service.config == config
        assert service._history == []


class TestChatServiceMethods:
    """Testes para métodos do ChatService."""
    
    @pytest.fixture
    def mock_service(self):
        """Serviço mockado para testes."""
        with patch('app.services.chat_service.SQLGenerator') as mock_gen, \
             patch('app.services.chat_service.QueryExecutor') as mock_exec, \
             patch('app.services.chat_service.ResponseFormatter') as mock_fmt:
            
            config = ChatServiceConfig(
                google_api_key="test-key",
                db_path="test.db"
            )
            
            service = ChatService(config)
            service._mock_generator = mock_gen
            service._mock_executor = mock_exec
            service._mock_formatter = mock_fmt
            
            yield service
    
    def test_get_suggestions(self, mock_service):
        """Testa lista de sugestões."""
        suggestions = mock_service.get_suggestions()
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)
    
    def test_get_history_empty(self, mock_service):
        """Testa histórico vazio."""
        history = mock_service.get_history()
        
        assert history == []
    
    def test_clear_history(self, mock_service):
        """Testa limpeza de histórico."""
        # Adiciona algo ao histórico
        mock_service._history = [{'question': 'test'}]
        
        mock_service.clear_history()
        
        assert mock_service._history == []
    
    def test_get_history_with_limit(self, mock_service):
        """Testa limite no histórico."""
        mock_service._history = [
            {'question': f'q{i}'} for i in range(20)
        ]
        
        history = mock_service.get_history(limit=5)
        
        assert len(history) == 5


class TestChatServiceChat:
    """Testes para o método chat."""
    
    @pytest.fixture
    def configured_service(self, temp_db):
        """Serviço configurado com mocks."""
        with patch('app.services.chat_service.SQLGenerator') as mock_gen, \
             patch('app.services.chat_service.ResponseFormatter') as mock_fmt:
            
            # Configura mock do generator
            mock_gen_instance = Mock()
            mock_gen.return_value = mock_gen_instance
            
            # Configura mock do formatter
            mock_fmt_instance = Mock()
            mock_fmt.return_value = mock_fmt_instance
            mock_fmt_instance.format.return_value = "Resposta formatada"
            mock_fmt_instance.format_conversation.return_value = "Resposta de conversa"
            mock_fmt_instance.format_error.return_value = "Erro formatado"
            
            config = ChatServiceConfig(
                google_api_key="test-key",
                db_path=temp_db
            )
            
            service = ChatService(config)
            service._mock_gen = mock_gen_instance
            service._mock_fmt = mock_fmt_instance
            
            yield service
    
    def test_chat_query_success(self, configured_service):
        """Testa chat com query SQL bem-sucedida."""
        from app.services.chat.core import SQLResult
        
        # Configura resposta do generator
        configured_service._mock_gen.generate.return_value = SQLResult(
            success=True,
            query="SELECT COUNT(*) as total FROM alunos",
            intent="query"
        )
        
        # Precisa sobrescrever o sql_generator
        configured_service.sql_generator = configured_service._mock_gen
        
        response = configured_service.chat("Quantos alunos temos?")
        
        assert isinstance(response, ChatResponse)
    
    def test_chat_conversation(self, configured_service):
        """Testa chat de conversa (sem SQL)."""
        from app.services.chat.core import SQLResult
        
        configured_service._mock_gen.generate.return_value = SQLResult(
            success=True,
            query=None,
            intent="conversation"
        )
        
        configured_service.sql_generator = configured_service._mock_gen
        
        response = configured_service.chat("Olá!")
        
        assert isinstance(response, ChatResponse)
        assert response.intent == "conversation"
    
    def test_chat_logs_interaction(self, configured_service):
        """Testa se chat registra interação no histórico."""
        from app.services.chat.core import SQLResult
        
        configured_service._mock_gen.generate.return_value = SQLResult(
            success=True,
            query=None,
            intent="conversation"
        )
        
        configured_service.sql_generator = configured_service._mock_gen
        
        initial_len = len(configured_service._history)
        configured_service.chat("Teste")
        
        assert len(configured_service._history) == initial_len + 1


class TestChatServiceHelperMethods:
    """Testes para métodos auxiliares do ChatService."""
    
    @pytest.fixture
    def service_with_db(self, temp_db):
        """Serviço com banco de dados real."""
        with patch('app.services.chat_service.SQLGenerator'), \
             patch('app.services.chat_service.ResponseFormatter'):
            
            config = ChatServiceConfig(
                google_api_key="test-key",
                db_path=temp_db
            )
            
            yield ChatService(config)
    
    def test_get_profile_summary(self, service_with_db):
        """Testa resumo de perfis."""
        summary = service_with_db.get_profile_summary()
        
        assert isinstance(summary, dict)
    
    def test_get_student_info_exists(self, service_with_db):
        """Testa busca de aluno existente."""
        info = service_with_db.get_student_info('RA-1')
        
        assert info is not None
        assert info['ra'] == 'RA-1'
    
    def test_get_student_info_not_exists(self, service_with_db):
        """Testa busca de aluno inexistente."""
        info = service_with_db.get_student_info('RA-INEXISTENTE')
        
        assert info is None
    
    def test_get_students_by_profile(self, service_with_db):
        """Testa busca por perfil."""
        students = service_with_db.get_students_by_profile('Destaque')
        
        assert isinstance(students, list)
    
    def test_get_students_by_profile_with_limit(self, service_with_db):
        """Testa busca por perfil com limite."""
        students = service_with_db.get_students_by_profile('Destaque', limit=1)
        
        assert len(students) <= 1
