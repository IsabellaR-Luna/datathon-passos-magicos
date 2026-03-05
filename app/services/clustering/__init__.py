# __init__.py
"""
Pacote de clustering para classificação de alunos.

Módulos:
    - core: Classes auxiliares (preprocessor, trainer, predictor, etc.)
    - cluster_service: Serviço principal (orquestrador)
"""

from app.services.clustering.core import (
    # Classes
    DataPreprocessor,
    ClusterTrainer,
    ClusterPredictor,
    ProfileMapper,
    ClusterEvaluator,
    # Dataclasses
    TrainerConfig,
    TrainingResult,
    PredictionResult,
    ValidationResult,
    DriftResult,
    EvaluationMetrics,
    # Exceções
    ClusteringError,
    PreprocessingError,
    TrainingError,
    PredictionError,
    ValidationError
)

__all__ = [
    # Classes
    'DataPreprocessor',
    'ClusterTrainer',
    'ClusterPredictor',
    'ProfileMapper',
    'ClusterEvaluator',
    # Dataclasses
    'TrainerConfig',
    'TrainingResult',
    'PredictionResult',
    'ValidationResult',
    'DriftResult',
    'EvaluationMetrics',
    # Exceções
    'ClusteringError',
    'PreprocessingError',
    'TrainingError',
    'PredictionError',
    'ValidationError'
]