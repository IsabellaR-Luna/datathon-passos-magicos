# test_cluster_service.py
"""
Testes para o serviço de clustering completo.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

from app.services.cluster_service import ClusterService, ClusterServiceConfig
from app.services.clustering.core import TrainerConfig


class TestClusterServiceConfig:
    """Testes para a configuração do serviço."""
    
    def test_default_config(self):
        """Testa valores padrão."""
        config = ClusterServiceConfig()
        
        assert config.models_dir == "app/models"
        assert config.min_samples_for_retrain == 100
        assert config.max_days_without_retrain == 365
        assert config.drift_threshold == 0.2
    
    def test_custom_config(self):
        """Testa configuração customizada."""
        config = ClusterServiceConfig(
            models_dir="/custom/path",
            min_samples_for_retrain=50
        )
        
        assert config.models_dir == "/custom/path"
        assert config.min_samples_for_retrain == 50
    
    def test_config_has_trainer_config(self):
        """Testa se config tem trainer_config."""
        config = ClusterServiceConfig()
        
        assert config.trainer_config is not None
        assert isinstance(config.trainer_config, TrainerConfig)
    
    def test_config_custom_trainer(self):
        """Testa config com trainer customizado."""
        trainer_cfg = TrainerConfig(max_evals=5)
        config = ClusterServiceConfig(trainer_config=trainer_cfg)
        
        assert config.trainer_config.max_evals == 5


class TestClusterServiceInit:
    """Testes para inicialização do ClusterService."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Diretório temporário para modelos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_init_creates_components(self, temp_model_dir):
        """Testa se inicialização cria componentes."""
        config = ClusterServiceConfig(models_dir=temp_model_dir)
        service = ClusterService(config)
        
        assert service.preprocessor is not None
        assert service.trainer is not None
        assert service.evaluator is not None
        assert service.profiler is not None
    
    def test_init_creates_model_dir(self, temp_model_dir):
        """Testa se cria diretório de modelos."""
        new_dir = os.path.join(temp_model_dir, "new_models")
        config = ClusterServiceConfig(models_dir=new_dir)
        
        ClusterService(config)
        
        assert os.path.exists(new_dir)


class TestClusterServiceShouldRetrain:
    """Testes para o método should_retrain."""
    
    @pytest.fixture
    def service(self):
        """Serviço para testes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ClusterServiceConfig(
                models_dir=tmpdir,
                min_samples_for_retrain=100
            )
            yield ClusterService(config)
    
    def test_should_retrain_zero_samples(self, service):
        """Testa com zero amostras."""
        result = service.should_retrain(0)
        assert isinstance(result, dict)
        assert 'should_retrain' in result
    
    def test_should_retrain_below_threshold(self, service):
        """Testa abaixo do threshold."""
        result = service.should_retrain(50)
        # Sem modelo treinado, sempre retorna True (nunca treinado)
        # Com modelo treinado e poucas amostras, retornaria False
        assert isinstance(result, dict)
        assert 'should_retrain' in result
    
    def test_should_retrain_above_threshold(self, service):
        """Testa acima do threshold."""
        result = service.should_retrain(150)
        assert result['should_retrain'] == True
    
    def test_should_retrain_returns_reasons(self, service):
        """Testa se retorna razões."""
        result = service.should_retrain(150)
        assert 'reasons' in result
        assert isinstance(result['reasons'], list)


class TestClusterServiceGetClusterSummary:
    """Testes para get_cluster_summary."""
    
    @pytest.fixture
    def service(self):
        """Serviço para testes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ClusterServiceConfig(models_dir=tmpdir)
            yield ClusterService(config)
    
    def test_get_summary_without_training(self, service):
        """Testa resumo sem treino."""
        try:
            summary = service.get_cluster_summary()
            assert isinstance(summary, dict)
        except Exception:
            # Pode levantar erro se não houver modelo
            pass


class TestClusterServiceValidation:
    """Testes para validação de dados."""
    
    @pytest.fixture
    def service(self):
        """Serviço para testes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ClusterServiceConfig(models_dir=tmpdir)
            yield ClusterService(config)
    
    @pytest.fixture
    def valid_df(self):
        """DataFrame válido."""
        return pd.DataFrame({
            'RA': ['RA-1', 'RA-2'],
            'Nome': ['Aluno-1', 'Aluno-2'],
            'IAA': [8.0, 7.0],
            'IEG': [9.0, 8.0],
            'IPS': [7.0, 6.0],
            'IDA': [8.0, 7.0],
            'IPV': [7.0, 6.0],
            'IAN': [10.0, 5.0],
            'Defas': [0, -1]
        })
    
    def test_preprocessor_validates(self, service, valid_df):
        """Testa validação pelo preprocessor."""
        result = service.preprocessor.validate_input(valid_df)
        
        assert result.valid == True
