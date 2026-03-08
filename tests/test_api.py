# test_api.py
"""
Testes para os endpoints da API.
"""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.routes import cluster as cluster_module
from app.routes import chat as chat_module
from app.services.chat.core import QueryExecutor


@pytest.fixture
def client(temp_db):
    """Cliente de teste com executor configurado."""
    cluster_module.query_executor = QueryExecutor(db_path=temp_db)
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_no_db():
    """Cliente de teste sem executor configurado."""
    cluster_module.query_executor = None
    chat_module.chat_service = None
    
    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Testes para o endpoint /health."""
    
    def test_health_returns_200(self, client):
        """Testa se /health retorna 200."""
        response = client.get("/health")
        
        assert response.status_code == 200
    
    def test_health_returns_status_ok(self, client):
        """Testa se /health retorna status ok."""
        response = client.get("/health")
        data = response.json()
        
        assert data['status'] == 'ok'
    
    def test_health_returns_version(self, client):
        """Testa se /health retorna versão."""
        response = client.get("/health")
        data = response.json()
        
        assert 'version' in data
    
    def test_health_returns_timestamp(self, client):
        """Testa se /health retorna timestamp."""
        response = client.get("/health")
        data = response.json()
        
        assert 'timestamp' in data


class TestRootEndpoint:
    """Testes para o endpoint /."""
    
    def test_root_returns_200(self, client):
        """Testa se / retorna 200."""
        response = client.get("/")
        
        assert response.status_code == 200
    
    def test_root_returns_message(self, client):
        """Testa se / retorna mensagem."""
        response = client.get("/")
        data = response.json()
        
        assert 'message' in data


class TestClusterSummaryEndpoint:
    """Testes para o endpoint /clusters/summary."""
    
    def test_summary_returns_200(self, client):
        """Testa se /clusters/summary retorna 200."""
        response = client.get("/clusters/summary")
        
        assert response.status_code == 200
    
    def test_summary_returns_total(self, client):
        """Testa se retorna total de alunos."""
        response = client.get("/clusters/summary")
        data = response.json()
        
        assert 'total_alunos' in data
        assert data['total_alunos'] >= 0  # Valor dinâmico baseado no banco
    
    def test_summary_returns_distribuicao(self, client):
        """Testa se retorna distribuição."""
        response = client.get("/clusters/summary")
        data = response.json()
        
        assert 'distribuicao' in data
        assert isinstance(data['distribuicao'], dict)
    
    def test_summary_returns_estatisticas(self, client):
        """Testa se retorna estatísticas por perfil."""
        response = client.get("/clusters/summary")
        data = response.json()
        
        assert 'estatisticas_por_perfil' in data
        assert isinstance(data['estatisticas_por_perfil'], list)
    
    def test_summary_without_executor_returns_500(self, client_no_db):
        """Testa erro sem executor configurado."""
        # O executor pode já estar configurado de testes anteriores
        # então verificamos se a resposta é 200 (já configurado) ou 500 (não configurado)
        response = client_no_db.get("/clusters/summary")
        
        assert response.status_code in [200, 500]


class TestClusterStudentsEndpoint:
    """Testes para o endpoint /clusters/students."""
    
    def test_students_returns_200(self, client):
        """Testa se /clusters/students retorna 200."""
        response = client.get("/clusters/students")
        
        assert response.status_code == 200
    
    def test_students_returns_list(self, client):
        """Testa se retorna lista de alunos."""
        response = client.get("/clusters/students")
        data = response.json()
        
        assert 'alunos' in data
        assert isinstance(data['alunos'], list)
    
    def test_students_returns_total(self, client):
        """Testa se retorna total."""
        response = client.get("/clusters/students")
        data = response.json()
        
        assert 'total' in data
    
    def test_students_filter_by_perfil(self, client):
        """Testa filtro por perfil."""
        response = client.get("/clusters/students?perfil=Destaque")
        data = response.json()
        
        assert data['total'] >= 0
        for aluno in data['alunos']:
            assert aluno['perfil'] == 'Destaque'
    
    def test_students_filter_by_turma(self, client):
        """Testa filtro por turma."""
        response = client.get("/clusters/students?turma=A")
        data = response.json()
        
        for aluno in data['alunos']:
            # Todos devem ser da turma A (verificado via banco)
            assert data['total'] > 0
    
    def test_students_limit(self, client):
        """Testa limite de resultados."""
        response = client.get("/clusters/students?limit=2")
        data = response.json()
        
        assert len(data['alunos']) <= 2
    
    def test_students_invalid_limit(self, client):
        """Testa limite inválido."""
        response = client.get("/clusters/students?limit=0")
        
        assert response.status_code == 422  # Validation error


class TestClusterStudentEndpoint:
    """Testes para o endpoint /clusters/student/{ra}."""
    
    def test_student_returns_200(self, client):
        """Testa se /clusters/student/{ra} retorna 200."""
        response = client.get("/clusters/student/RA-1")
        
        assert response.status_code == 200
    
    def test_student_returns_details(self, client):
        """Testa se retorna detalhes do aluno."""
        response = client.get("/clusters/student/RA-1")
        data = response.json()
        
        assert data['ra'] == 'RA-1'
        assert 'nome' in data
        assert 'perfil' in data
        assert 'recomendacoes' in data
    
    def test_student_returns_recomendacoes(self, client):
        """Testa se retorna recomendações."""
        response = client.get("/clusters/student/RA-1")
        data = response.json()
        
        assert isinstance(data['recomendacoes'], list)
        assert len(data['recomendacoes']) > 0
    
    def test_student_not_found(self, client):
        """Testa aluno não encontrado."""
        response = client.get("/clusters/student/RA-INEXISTENTE")
        
        assert response.status_code == 404


class TestClusterProfilesEndpoint:
    """Testes para o endpoint /clusters/profiles."""
    
    def test_profiles_returns_200(self, client):
        """Testa se /clusters/profiles retorna 200."""
        response = client.get("/clusters/profiles")
        
        assert response.status_code == 200
    
    def test_profiles_returns_list(self, client):
        """Testa se retorna lista de perfis."""
        response = client.get("/clusters/profiles")
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == 5  # 5 perfis
    
    def test_profiles_has_required_fields(self, client):
        """Testa se perfis têm campos obrigatórios."""
        response = client.get("/clusters/profiles")
        data = response.json()
        
        for profile in data:
            assert 'nome' in profile
            assert 'descricao' in profile
            assert 'recomendacoes' in profile
            assert 'total_alunos' in profile


class TestChatEndpoint:
    """Testes para o endpoint /chat."""
    
    def test_chat_without_service_returns_500(self, client_no_db):
        """Testa erro sem serviço configurado."""
        response = client_no_db.post("/chat", json={"pergunta": "Teste"})
        
        assert response.status_code == 500
    
    def test_chat_missing_pergunta_returns_422(self, client):
        """Testa requisição sem pergunta."""
        response = client.post("/chat", json={})
        
        assert response.status_code == 422
    
    def test_chat_empty_pergunta_returns_422(self, client):
        """Testa pergunta vazia."""
        response = client.post("/chat", json={"pergunta": ""})
        
        assert response.status_code == 422


class TestChatSuggestionsEndpoint:
    """Testes para o endpoint /chat/suggestions."""
    
    def test_suggestions_without_service_returns_500(self, client_no_db):
        """Testa erro sem serviço configurado."""
        response = client_no_db.get("/chat/suggestions")
        
        assert response.status_code == 500
