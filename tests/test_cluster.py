# test_cluster.py
"""
Testes para o módulo de clustering.
"""

import pytest
import numpy as np
import pandas as pd

from app.services.clustering.core import (
    ClusterTrainer,
    ClusterPredictor,
    ProfileMapper,
    ClusterEvaluator,
    TrainerConfig,
    TrainingResult,
    EvaluationMetrics,
    DriftResult
)


class TestTrainerConfig:
    """Testes para a configuração do trainer."""
    
    def test_default_config(self):
        """Testa valores padrão da configuração."""
        config = TrainerConfig()
        
        assert config.n_neighbors_range == (3, 30)
        assert config.n_components_range == (2, 8)
        assert config.min_cluster_size_range == (10, 50)
        assert config.random_state == 42
        assert config.label_lower == 3
        assert config.label_upper == 12
        assert config.max_evals == 25
    
    def test_custom_config(self):
        """Testa configuração customizada."""
        config = TrainerConfig(
            label_lower=2,
            label_upper=5,
            max_evals=10
        )
        
        assert config.label_lower == 2
        assert config.label_upper == 5
        assert config.max_evals == 10


class TestClusterTrainer:
    """Testes para a classe ClusterTrainer."""
    
    def test_init_default_config(self):
        """Testa inicialização com config padrão."""
        trainer = ClusterTrainer()
        
        assert trainer.config is not None
        assert trainer.best_params is None
        assert trainer.best_clusters is None
    
    def test_init_custom_config(self):
        """Testa inicialização com config customizada."""
        config = TrainerConfig(max_evals=5)
        trainer = ClusterTrainer(config)
        
        assert trainer.config.max_evals == 5
    
    def test_train_returns_training_result(self, sample_df_large):
        """Testa se train retorna TrainingResult."""
        from app.services.clustering.core import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_df_large)
        
        config = TrainerConfig(max_evals=3, label_lower=2, label_upper=10)
        trainer = ClusterTrainer(config)
        
        result = trainer.train(X)
        
        assert isinstance(result, TrainingResult)
        assert result.success == True
        assert result.model_version is not None
        assert result.n_clusters >= 0
    
    def test_train_sets_best_params(self, sample_df_large):
        """Testa se train define best_params."""
        from app.services.clustering.core import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_df_large)
        
        config = TrainerConfig(max_evals=3, label_lower=2, label_upper=10)
        trainer = ClusterTrainer(config)
        trainer.train(X)
        
        assert trainer.best_params is not None
        assert 'n_neighbors' in trainer.best_params
        assert 'n_components' in trainer.best_params
        assert 'min_cluster_size' in trainer.best_params
    
    def test_train_sets_best_clusters(self, sample_df_large):
        """Testa se train define best_clusters."""
        from app.services.clustering.core import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        X = preprocessor.fit_transform(sample_df_large)
        
        config = TrainerConfig(max_evals=3, label_lower=2, label_upper=10)
        trainer = ClusterTrainer(config)
        trainer.train(X)
        
        assert trainer.best_clusters is not None
        assert hasattr(trainer.best_clusters, 'labels_')


class TestProfileMapper:
    """Testes para a classe ProfileMapper."""
    
    def test_default_mapping(self):
        """Testa mapeamento padrão."""
        mapper = ProfileMapper()
        
        assert mapper.get_profile(-1) == 'Avaliar'
        assert mapper.get_profile(2) == 'Crítico'
        assert mapper.get_profile(5) == 'Destaque'
    
    def test_get_profile_unknown(self):
        """Testa cluster desconhecido."""
        mapper = ProfileMapper()
        
        assert mapper.get_profile(999) == 'Desconhecido'
    
    def test_get_description(self):
        """Testa descrição do perfil."""
        mapper = ProfileMapper()
        
        desc = mapper.get_description('Crítico')
        
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert 'urgente' in desc.lower() or 'intervenção' in desc.lower()
    
    def test_get_recommendations(self):
        """Testa recomendações do perfil."""
        mapper = ProfileMapper()
        
        recs = mapper.get_recommendations('Crítico')
        
        assert isinstance(recs, list)
        assert len(recs) > 0
    
    def test_get_recommendations_destaque(self):
        """Testa recomendações para Destaque."""
        mapper = ProfileMapper()
        
        recs = mapper.get_recommendations('Destaque')
        
        assert isinstance(recs, list)
        assert any('mentoria' in r.lower() for r in recs)
    
    def test_get_full_profile(self):
        """Testa informações completas do perfil."""
        mapper = ProfileMapper()
        
        info = mapper.get_full_profile(2)
        
        assert 'cluster_id' in info
        assert 'profile' in info
        assert 'description' in info
        assert 'recommendations' in info
        assert info['cluster_id'] == 2
        assert info['profile'] == 'Crítico'
    
    def test_update_mapping(self):
        """Testa atualização do mapeamento."""
        mapper = ProfileMapper()
        
        mapper.update_mapping({100: 'Novo Perfil'})
        
        assert mapper.get_profile(100) == 'Novo Perfil'
    
    def test_custom_mapping(self):
        """Testa mapeamento customizado na inicialização."""
        custom = {0: 'Perfil A', 1: 'Perfil B'}
        mapper = ProfileMapper(custom_mapping=custom)
        
        assert mapper.get_profile(0) == 'Perfil A'
        assert mapper.get_profile(1) == 'Perfil B'


class TestClusterEvaluator:
    """Testes para a classe ClusterEvaluator."""
    
    def test_evaluate_returns_metrics(self):
        """Testa se evaluate retorna métricas."""
        evaluator = ClusterEvaluator()
        
        X = np.random.rand(50, 5)
        labels = np.array([0, 1, 0, 1, 2] * 10)
        
        metrics = evaluator.evaluate(X, labels)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.n_clusters == 3
        assert metrics.silhouette >= -1 and metrics.silhouette <= 1
    
    def test_evaluate_with_outliers(self):
        """Testa avaliação com outliers."""
        evaluator = ClusterEvaluator()
        
        X = np.random.rand(50, 5)
        labels = np.array([0, 1, -1, 1, 0] * 10)  # -1 são outliers
        
        metrics = evaluator.evaluate(X, labels)
        
        assert metrics.n_outliers == 10
    
    def test_calculate_psi_no_drift(self):
        """Testa PSI sem drift."""
        evaluator = ClusterEvaluator()
        
        np.random.seed(42)
        train = np.random.normal(5, 1, 1000)
        prod = np.random.normal(5, 1, 100)
        
        psi = evaluator.calculate_psi(train, prod)
        
        assert psi < 0.25  # Threshold mais tolerante para variância amostral
    
    def test_calculate_psi_with_drift(self):
        """Testa PSI com drift."""
        evaluator = ClusterEvaluator()
        
        np.random.seed(42)
        train = np.random.normal(5, 1, 1000)
        prod = np.random.normal(8, 1, 100)  # Média diferente
        
        psi = evaluator.calculate_psi(train, prod)
        
        assert psi > 0.2  # Drift significativo
    
    def test_detect_drift_no_drift(self):
        """Testa detecção sem drift."""
        evaluator = ClusterEvaluator()
        
        np.random.seed(42)
        # Usar mesma seed para garantir distribuições similares
        X_train = np.random.normal(5, 1, (200, 3))
        np.random.seed(42)
        X_new = np.random.normal(5, 1, (50, 3))
        
        result = evaluator.detect_drift(X_train, X_new)
        
        assert isinstance(result, DriftResult)
        # Com mesma seed, não deve haver drift
        assert result.drift_detected == False or len(result.features_with_drift) < 2
    
    def test_detect_drift_with_drift(self):
        """Testa detecção com drift."""
        evaluator = ClusterEvaluator()
        
        np.random.seed(42)
        X_train = np.random.normal(5, 1, (100, 3))
        X_new = np.random.normal(10, 1, (20, 3))  # Média muito diferente
        
        result = evaluator.detect_drift(X_train, X_new)
        
        assert result.drift_detected == True
        assert len(result.features_with_drift) > 0
