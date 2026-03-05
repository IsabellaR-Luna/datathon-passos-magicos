# cluster_service.py
"""
Serviço principal de clustering para classificação de alunos.

Este serviço orquestra todas as operações de clustering:
- Treinar modelo
- Fazer predições
- Obter estatísticas
- Retreinar quando necessário
- Detectar drift

Uso:
    service = ClusterService(config)
    
    # Treino
    result = service.train(df)
    
    # Predição
    prediction = service.predict({"IAA": 7.5, "IEG": 8.0, ...})
    
    # Estatísticas
    summary = service.get_cluster_summary()
"""

import logging
import joblib
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime

import numpy as np
import pandas as pd

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
    PredictionError
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
# CONFIGURAÇÃO DO SERVIÇO
# ============================================================================

@dataclass
class ClusterServiceConfig:
    """Configuração do serviço de clustering."""
    
    # Paths
    models_dir: str = "app/models"
    data_dir: str = "data"
    
    # Retreino
    min_samples_for_retrain: int = 100
    max_days_without_retrain: int = 365
    drift_threshold: float = 0.2
    
    # Trainer
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    
    # Modelo atual
    current_model_version: Optional[str] = None


# ============================================================================
# CLASSE PRINCIPAL: ClusterService
# ============================================================================

class ClusterService:
    """
    Serviço principal de clustering.
    
    Orquestra todas as operações:
    - Treinar modelo
    - Fazer predições
    - Obter estatísticas
    - Retreinar quando necessário
    """
    
    def __init__(self, config: Optional[ClusterServiceConfig] = None):
        self.config = config or ClusterServiceConfig()
        
        # Componentes
        self.preprocessor = DataPreprocessor()
        self.trainer = ClusterTrainer(self.config.trainer_config)
        self.predictor: Optional[ClusterPredictor] = None
        self.profiler = ProfileMapper()
        self.evaluator = ClusterEvaluator()
        
        # Estado
        self._is_trained = False
        self._training_data: Optional[np.ndarray] = None
        self._training_labels: Optional[np.ndarray] = None
        self._current_version: Optional[str] = None
        self._accumulated_samples: int = 0
        self._last_train_date: Optional[datetime] = None
        
        # Cria diretórios
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Tenta carregar modelo existente
        self._try_load_latest_model()
    
    # ========================================================================
    # TREINO
    # ========================================================================
    
    def train(self, df: pd.DataFrame) -> TrainingResult:
        """
        Pipeline completo de treino.
        
        Args:
            df: DataFrame com dados dos alunos
            
        Returns:
            TrainingResult com métricas e status
        """
        logger.info("=" * 70)
        logger.info("[CLUSTER SERVICE] INICIANDO PIPELINE DE TREINAMENTO")
        logger.info("=" * 70)
        logger.info(f"[CLUSTER SERVICE] Amostras recebidas: {len(df)}")
        
        try:
            # 1. Pré-processamento
            logger.info("[CLUSTER SERVICE] Etapa 1/4: Pré-processamento")
            X = self.preprocessor.fit_transform(df)
            
            # 2. Treino
            logger.info("[CLUSTER SERVICE] Etapa 2/4: Treinamento do modelo")
            result = self.trainer.train(X)
            
            if not result.success:
                logger.error(f"[CLUSTER SERVICE] Falha no treinamento: {result.error_message}")
                return result
            
            # 3. Configura predictor
            logger.info("[CLUSTER SERVICE] Etapa 3/4: Configurando predictor")
            self.predictor = ClusterPredictor(
                clusters=self.trainer.best_clusters,
                umap_model=self.trainer.umap_model
            )
            
            # 4. Salva modelo
            logger.info("[CLUSTER SERVICE] Etapa 4/4: Salvando modelo")
            self._save_model(result.model_version)
            
            # Atualiza estado
            self._is_trained = True
            self._training_data = X
            self._training_labels = self.trainer.best_clusters.labels_
            self._current_version = result.model_version
            self._last_train_date = datetime.now()
            self._accumulated_samples = 0
            
            logger.info("=" * 70)
            logger.info("[CLUSTER SERVICE] PIPELINE DE TREINAMENTO CONCLUÍDO")
            logger.info(f"[CLUSTER SERVICE] Versão do modelo: {result.model_version}")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"[CLUSTER SERVICE] ERRO NO PIPELINE: {str(e)}")
            raise TrainingError(f"Falha no treinamento: {str(e)}")
    
    def should_retrain(self, new_samples: int = 0) -> Dict[str, Any]:
        """
        Decide se deve retreinar o modelo.
        
        Args:
            new_samples: Número de novas amostras acumuladas
            
        Returns:
            Dict com decisão e motivo
        """
        logger.info("[CLUSTER SERVICE] Verificando necessidade de retreino...")
        
        self._accumulated_samples += new_samples
        
        reasons = []
        should_retrain = False
        
        # Critério 1: Volume de novos dados
        if self._accumulated_samples >= self.config.min_samples_for_retrain:
            reasons.append(f"Acumulou {self._accumulated_samples} novas amostras (mínimo: {self.config.min_samples_for_retrain})")
            should_retrain = True
        
        # Critério 2: Tempo desde último treino
        if self._last_train_date:
            days_since_train = (datetime.now() - self._last_train_date).days
            if days_since_train >= self.config.max_days_without_retrain:
                reasons.append(f"{days_since_train} dias desde último treino (máximo: {self.config.max_days_without_retrain})")
                should_retrain = True
        else:
            reasons.append("Modelo nunca foi treinado")
            should_retrain = True
        
        result = {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'accumulated_samples': self._accumulated_samples,
            'days_since_train': (datetime.now() - self._last_train_date).days if self._last_train_date else None,
            'current_version': self._current_version
        }
        
        logger.info(f"[CLUSTER SERVICE] Decisão de retreino: {should_retrain}")
        if reasons:
            logger.info(f"[CLUSTER SERVICE] Motivos: {reasons}")
        
        return result
    
    def retrain(self, df: pd.DataFrame) -> TrainingResult:
        """
        Retreina modelo com novos dados.
        
        Args:
            df: DataFrame com todos os dados (históricos + novos)
            
        Returns:
            TrainingResult
        """
        logger.info("[CLUSTER SERVICE] Iniciando RETREINO do modelo...")
        
        # Salva versão anterior como backup
        if self._current_version:
            self._backup_current_model()
        
        # Treina novo modelo
        result = self.train(df)
        
        if result.success:
            logger.info("[CLUSTER SERVICE] Retreino concluído com sucesso")
            self._accumulated_samples = 0
        else:
            logger.warning("[CLUSTER SERVICE] Retreino falhou, mantendo modelo anterior")
            self._restore_backup()
        
        return result
    
    # ========================================================================
    # INFERÊNCIA
    # ========================================================================
    
    def predict(self, student_data: Dict[str, float]) -> PredictionResult:
        """
        Prediz perfil de um aluno.
        
        Args:
            student_data: Dict com métricas do aluno
            
        Returns:
            PredictionResult com perfil e recomendações
        """
        if not self._is_trained:
            raise PredictionError("Modelo não treinado. Execute train() primeiro.")
        
        logger.info(f"[CLUSTER SERVICE] Predizendo perfil para aluno...")
        
        try:
            # Prepara dados
            X = self.preprocessor.transform_single(student_data)
            
            # Prediz
            cluster_id, confidence = self.predictor.predict_single(X)
            
            # Mapeia para perfil
            profile_info = self.profiler.get_full_profile(cluster_id)
            
            result = PredictionResult(
                cluster_id=cluster_id,
                profile=profile_info['profile'],
                confidence=confidence,
                profile_description=profile_info['description'],
                recommendations=profile_info['recommendations']
            )
            
            logger.info(f"[CLUSTER SERVICE] Predição: {result.profile} (confiança: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"[CLUSTER SERVICE] Erro na predição: {str(e)}")
            raise PredictionError(f"Falha na predição: {str(e)}")
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prediz perfil de múltiplos alunos.
        
        Args:
            df: DataFrame com dados dos alunos
            
        Returns:
            DataFrame original + colunas de perfil
        """
        if not self._is_trained:
            raise PredictionError("Modelo não treinado. Execute train() primeiro.")
        
        logger.info(f"[CLUSTER SERVICE] Predição em lote para {len(df)} alunos...")
        
        try:
            # Prepara dados
            X = self.preprocessor.transform(df)
            
            # Prediz
            labels, probabilities = self.predictor.predict(X)
            
            # Adiciona ao DataFrame
            df_result = df.copy()
            df_result['cluster_id'] = labels
            df_result['cluster_confidence'] = probabilities
            df_result['perfil'] = df_result['cluster_id'].apply(self.profiler.get_profile)
            
            logger.info(f"[CLUSTER SERVICE] Predição em lote concluída")
            
            return df_result
            
        except Exception as e:
            logger.error(f"[CLUSTER SERVICE] Erro na predição em lote: {str(e)}")
            raise PredictionError(f"Falha na predição em lote: {str(e)}")
    
    # ========================================================================
    # ESTATÍSTICAS
    # ========================================================================
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """
        Retorna estatísticas agregadas dos clusters.
        
        Returns:
            Dict com resumo dos clusters
        """
        if not self._is_trained or self._training_labels is None:
            raise ClusteringError("Modelo não treinado.")
        
        logger.info("[CLUSTER SERVICE] Gerando resumo dos clusters...")
        
        # Avalia clustering
        metrics = self.evaluator.evaluate(
            self._training_data,
            self._training_labels,
            self.trainer.best_clusters.probabilities_ if self.trainer.best_clusters else None
        )
        
        # Agrupa por perfil
        profile_counts = {}
        for cluster_id, count in metrics.cluster_distribution.items():
            profile = self.profiler.get_profile(cluster_id)
            profile_counts[profile] = profile_counts.get(profile, 0) + count
        
        summary = {
            'model_version': self._current_version,
            'last_train_date': self._last_train_date.isoformat() if self._last_train_date else None,
            'total_samples': int(len(self._training_labels)),
            'metrics': {
                'silhouette_score': round(metrics.silhouette, 4),
                'n_clusters': metrics.n_clusters,
                'n_outliers': metrics.n_outliers,
                'low_confidence_ratio': round(metrics.low_confidence_ratio, 4)
            },
            'cluster_distribution': metrics.cluster_distribution,
            'profile_distribution': profile_counts,
            'profiles': {
                profile: {
                    'count': profile_counts.get(profile, 0),
                    'description': self.profiler.get_description(profile),
                    'recommendations': self.profiler.get_recommendations(profile)
                }
                for profile in ['Crítico', 'Atenção', 'Em Desenvolvimento', 'Destaque', 'Avaliar']
            }
        }
        
        logger.info(f"[CLUSTER SERVICE] Resumo gerado: {metrics.n_clusters} clusters, {len(self._training_labels)} amostras")
        
        return summary
    
    def get_students_by_profile(
        self,
        df: pd.DataFrame,
        profile: str
    ) -> pd.DataFrame:
        """
        Retorna alunos de um perfil específico.
        
        Args:
            df: DataFrame com dados dos alunos (já com cluster atribuído)
            profile: Nome do perfil desejado
            
        Returns:
            DataFrame filtrado
        """
        logger.info(f"[CLUSTER SERVICE] Filtrando alunos do perfil: {profile}")
        
        # Se não tem coluna de perfil, faz predição
        if 'perfil' not in df.columns:
            df = self.predict_batch(df)
        
        return df[df['perfil'] == profile]
    
    # ========================================================================
    # MONITORAMENTO
    # ========================================================================
    
    def check_drift(self, new_data: pd.DataFrame) -> DriftResult:
        """
        Verifica se há drift nos dados.
        
        Args:
            new_data: Novos dados para comparar
            
        Returns:
            DriftResult
        """
        if self._training_data is None:
            raise ClusteringError("Sem dados de treino para comparação.")
        
        logger.info("[CLUSTER SERVICE] Verificando drift nos dados...")
        
        # Prepara novos dados
        X_new = self.preprocessor.transform(new_data)
        
        # Detecta drift
        drift_result = self.evaluator.detect_drift(
            self._training_data,
            X_new,
            feature_names=DataPreprocessor.FEATURE_COLUMNS
        )
        
        # Atualiza decisão de retreino
        if drift_result.drift_detected:
            logger.warning("[CLUSTER SERVICE] DRIFT DETECTADO - Considere retreinar o modelo")
        
        return drift_result
    
    # ========================================================================
    # PERSISTÊNCIA
    # ========================================================================
    
    def _save_model(self, version: str):
        """Salva modelo e componentes."""
        logger.info(f"[CLUSTER SERVICE] Salvando modelo versão {version}...")
        
        model_dir = Path(self.config.models_dir) / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva componentes
        joblib.dump(self.trainer.best_clusters, model_dir / "hdbscan_model.joblib")
        joblib.dump(self.trainer.umap_model, model_dir / "umap_model.joblib")
        joblib.dump(self.preprocessor.scaler, model_dir / "scaler.joblib")
        
        # Salva metadados
        metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'best_params': self.trainer.best_params,
            'preprocessor_state': self.preprocessor.get_state(),
            'cluster_mapping': self.profiler.cluster_to_profile,
            'n_samples': len(self._training_labels) if self._training_labels is not None else 0
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Atualiza link para modelo atual
        current_link = Path(self.config.models_dir) / "current_version.txt"
        with open(current_link, 'w') as f:
            f.write(version)
        
        logger.info(f"[CLUSTER SERVICE] Modelo salvo em {model_dir}")
    
    def _try_load_latest_model(self):
        """Tenta carregar o modelo mais recente."""
        current_link = Path(self.config.models_dir) / "current_version.txt"
        
        if not current_link.exists():
            logger.info("[CLUSTER SERVICE] Nenhum modelo anterior encontrado")
            return
        
        try:
            with open(current_link, 'r') as f:
                version = f.read().strip()
            
            self._load_model(version)
            logger.info(f"[CLUSTER SERVICE] Modelo {version} carregado com sucesso")
            
        except Exception as e:
            logger.warning(f"[CLUSTER SERVICE] Falha ao carregar modelo: {str(e)}")
    
    def _load_model(self, version: str):
        """Carrega modelo de uma versão específica."""
        model_dir = Path(self.config.models_dir) / version
        
        if not model_dir.exists():
            raise ClusteringError(f"Versão {version} não encontrada")
        
        logger.info(f"[CLUSTER SERVICE] Carregando modelo versão {version}...")
        
        # Carrega componentes
        clusters = joblib.load(model_dir / "hdbscan_model.joblib")
        umap_model = joblib.load(model_dir / "umap_model.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        
        # Carrega metadados
        with open(model_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Configura componentes
        self.preprocessor.scaler = scaler
        self.preprocessor.load_state(metadata.get('preprocessor_state', {}))
        
        self.predictor = ClusterPredictor(clusters=clusters, umap_model=umap_model)
        
        self.trainer.best_clusters = clusters
        self.trainer.umap_model = umap_model
        self.trainer.best_params = metadata.get('best_params', {})
        
        if metadata.get('cluster_mapping'):
            # Converte keys de string para int
            mapping = {int(k): v for k, v in metadata['cluster_mapping'].items()}
            self.profiler.update_mapping(mapping)
        
        # Atualiza estado
        self._is_trained = True
        self._current_version = version
        self._training_labels = clusters.labels_
        
        logger.info(f"[CLUSTER SERVICE] Modelo {version} carregado")
    
    def _backup_current_model(self):
        """Faz backup do modelo atual antes de retreinar."""
        if self._current_version:
            backup_name = f"{self._current_version}_backup"
            logger.info(f"[CLUSTER SERVICE] Criando backup: {backup_name}")
            # Implementar cópia do diretório se necessário
    
    def _restore_backup(self):
        """Restaura modelo do backup se retreino falhar."""
        logger.info("[CLUSTER SERVICE] Restaurando modelo do backup...")
        # Implementar restauração se necessário
    
    # ========================================================================
    # PROPRIEDADES
    # ========================================================================
    
    @property
    def is_trained(self) -> bool:
        """Retorna se o modelo está treinado."""
        return self._is_trained
    
    @property
    def current_version(self) -> Optional[str]:
        """Retorna versão atual do modelo."""
        return self._current_version
    
    @property
    def accumulated_samples(self) -> int:
        """Retorna número de amostras acumuladas desde último treino."""
        return self._accumulated_samples



if __name__ == "__main__":
    # Exemplo de uso do serviço
    
    # Dados de exemplo
    df = pd.DataFrame({
        'IAA': [8.3, 0.0, 7.5, 9.0, 4.2, 8.8, 0.0, 6.5, 9.5, 3.1],
        'IEG': [4.1, 7.9, 8.0, 9.4, 5.5, 8.2, 4.0, 7.2, 9.0, 6.8],
        'IPS': [5.6, 5.6, 6.5, 7.5, 6.0, 7.5, 6.9, 5.8, 7.5, 5.2],
        'IDA': [4.0, 5.6, 6.8, 9.3, 3.5, 8.0, 1.5, 5.2, 8.5, 4.8],
        'IPV': [7.2, 7.5, 6.7, 7.2, 5.8, 6.8, 6.5, 6.0, 8.0, 5.5],
        'IAN': [5.0, 10.0, 10.0, 5.0, 5.0, 10.0, 5.0, 5.0, 10.0, 5.0],
        'Defas': [-1, 0, 0, -2, -1, 0, 0, -1, -1, -2]
    })
    
    # Cria serviço
    config = ClusterServiceConfig()
    service = ClusterService(config)
    
    # Treina
    print("\n" + "="*70)
    print("TESTE DO CLUSTER SERVICE")
    print("="*70)
    
    result = service.train(df)
    print(f"\nResultado do treino: {result}")
    
    # Predição individual
    student = {
        'IAA': 7.5, 'IEG': 8.0, 'IPS': 6.5,
        'IDA': 6.0, 'IPV': 7.0, 'IAN': 10.0, 'Defas': 0
    }
    
    prediction = service.predict(student)
    print(f"\nPredição: {prediction}")
    
    # Resumo
    summary = service.get_cluster_summary()
    print(f"\nResumo: {summary}")