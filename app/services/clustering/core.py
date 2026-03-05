# core.py
"""
Classes auxiliares para o serviço de clustering.

Classes:
    - DataPreprocessor: Limpeza e preparação de dados
    - ClusterTrainer: Treino do modelo (UMAP + HDBSCAN)
    - ClusterPredictor: Inferência em produção
    - ProfileMapper: Mapeamento cluster → perfil
    - ClusterEvaluator: Métricas de avaliação
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
from datetime import datetime

import umap
import hdbscan
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASSES DE CONFIGURAÇÃO E RESULTADO
# ============================================================================

@dataclass
class TrainerConfig:
    """Configuração para o treinamento do modelo."""
    n_neighbors_range: Tuple[int, int] = (3, 30)
    n_components_range: Tuple[int, int] = (2, 8)
    min_cluster_size_range: Tuple[int, int] = (10, 50)
    min_samples_range: Tuple[int, int] = (2, 12)
    random_state: int = 42
    label_lower: int = 3
    label_upper: int = 12
    max_evals: int = 25
    prob_threshold: float = 0.05


@dataclass
class TrainingResult:
    """Resultado do treinamento."""
    success: bool
    model_version: str
    best_params: Dict[str, Any]
    n_clusters: int
    silhouette: float
    cluster_distribution: Dict[int, int]
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None


@dataclass
class PredictionResult:
    """Resultado de uma predição."""
    cluster_id: int
    profile: str
    confidence: float
    profile_description: str
    recommendations: List[str]


@dataclass
class ValidationResult:
    """Resultado da validação de dados."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DriftResult:
    """Resultado da detecção de drift."""
    drift_detected: bool
    psi_scores: Dict[str, float]
    features_with_drift: List[str]
    recommendation: str


@dataclass
class EvaluationMetrics:
    """Métricas de avaliação do clustering."""
    silhouette: float
    n_clusters: int
    n_outliers: int
    cluster_distribution: Dict[int, int]
    low_confidence_ratio: float


# ============================================================================
# EXCEÇÕES CUSTOMIZADAS
# ============================================================================

class ClusteringError(Exception):
    """Erro base para operações de clustering."""
    pass


class PreprocessingError(ClusteringError):
    """Erro durante pré-processamento."""
    pass


class TrainingError(ClusteringError):
    """Erro durante treinamento."""
    pass


class PredictionError(ClusteringError):
    """Erro durante predição."""
    pass


class ValidationError(ClusteringError):
    """Erro de validação de dados."""
    pass


# ============================================================================
# CLASSE: DataPreprocessor
# ============================================================================

class DataPreprocessor:
    """
    Prepara dados para clustering.
    
    Responsabilidades:
    - Selecionar features relevantes
    - Tratar valores nulos
    - Normalizar dados
    - Validar formato de entrada
    """
    
    FEATURE_COLUMNS = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'Defas']
    
    EXPECTED_RANGES = {
        'IAA': (0, 10),
        'IEG': (0, 10),
        'IPS': (0, 10),
        'IDA': (0, 10),
        'IPV': (0, 10),
        'IAN': (0, 10),
        'Defas': (-5, 5)
    }
    
    def __init__(self):
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._training_medians: Optional[Dict[str, float]] = None
        
    def validate_input(self, df: pd.DataFrame) -> ValidationResult:
        """Valida se o DataFrame tem as colunas e tipos corretos."""
        logger.info("[PREPROCESSING] Validando dados de entrada...")
        
        errors = []
        warnings = []
        
        # Verifica colunas obrigatórias
        missing_cols = set(self.FEATURE_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Colunas faltando: {missing_cols}")
        
        # Verifica valores nulos
        for col in self.FEATURE_COLUMNS:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    null_pct = (null_count / len(df)) * 100
                    warnings.append(f"{col}: {null_count} valores nulos ({null_pct:.1f}%)")
        
        # Verifica ranges
        for col, (min_val, max_val) in self.EXPECTED_RANGES.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    warnings.append(f"{col}: {out_of_range} valores fora do range [{min_val}, {max_val}]")
        
        valid = len(errors) == 0
        
        if valid:
            logger.info("[PREPROCESSING] Validação concluída: DADOS VÁLIDOS")
        else:
            logger.warning(f"[PREPROCESSING] Validação concluída: {len(errors)} erros encontrados")
        
        if warnings:
            logger.warning(f"[PREPROCESSING] {len(warnings)} avisos: {warnings}")
        
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Ajusta o scaler e transforma os dados (para treino).
        
        Args:
            df: DataFrame com os dados de treino
            
        Returns:
            Array normalizado pronto para clustering
        """
        logger.info("[PREPROCESSING] Iniciando fit_transform...")
        
        # Valida entrada
        validation = self.validate_input(df)
        if not validation.valid:
            raise PreprocessingError(f"Dados inválidos: {validation.errors}")
        
        # Seleciona features
        X = df[self.FEATURE_COLUMNS].copy()
        
        # Salva medianas para uso futuro
        self._training_medians = X.median().to_dict()
        logger.info(f"[PREPROCESSING] Medianas calculadas: {self._training_medians}")
        
        # Trata nulos com mediana
        X = X.fillna(X.median())
        
        # Normaliza
        X_scaled = self.scaler.fit_transform(X)
        self._is_fitted = True
        
        logger.info(f"[PREPROCESSING] fit_transform concluído - Shape: {X_scaled.shape}")
        
        return X_scaled
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforma dados usando scaler já ajustado (para inferência).
        
        Args:
            df: DataFrame com novos dados
            
        Returns:
            Array normalizado
        """
        if not self._is_fitted:
            raise PreprocessingError("Preprocessor não foi ajustado. Execute fit_transform primeiro.")
        
        logger.info("[PREPROCESSING] Iniciando transform...")
        
        # Seleciona features
        X = df[self.FEATURE_COLUMNS].copy()
        
        # Trata nulos com medianas do treino
        if self._training_medians:
            X = X.fillna(self._training_medians)
        else:
            X = X.fillna(X.median())
        
        # Normaliza com scaler existente
        X_scaled = self.scaler.transform(X)
        
        logger.info(f"[PREPROCESSING] transform concluído - Shape: {X_scaled.shape}")
        
        return X_scaled
    
    def transform_single(self, features: Dict[str, float]) -> np.ndarray:
        """
        Transforma um único registro para predição.
        
        Args:
            features: Dicionário com valores das features
            
        Returns:
            Array normalizado (1, n_features)
        """
        if not self._is_fitted:
            raise PreprocessingError("Preprocessor não foi ajustado. Execute fit_transform primeiro.")
        
        # Cria DataFrame de uma linha
        df = pd.DataFrame([features])
        
        return self.transform(df)
    
    def get_state(self) -> Dict[str, Any]:
        """Retorna estado do preprocessor para serialização."""
        return {
            'scaler_mean': self.scaler.mean_.tolist() if self._is_fitted else None,
            'scaler_scale': self.scaler.scale_.tolist() if self._is_fitted else None,
            'training_medians': self._training_medians,
            'is_fitted': self._is_fitted
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Carrega estado do preprocessor."""
        if state.get('scaler_mean') and state.get('scaler_scale'):
            self.scaler.mean_ = np.array(state['scaler_mean'])
            self.scaler.scale_ = np.array(state['scaler_scale'])
            self.scaler.var_ = self.scaler.scale_ ** 2
            self.scaler.n_features_in_ = len(state['scaler_mean'])
        self._training_medians = state.get('training_medians')
        self._is_fitted = state.get('is_fitted', False)


# ============================================================================
# CLASSE: ClusterTrainer
# ============================================================================

class ClusterTrainer:
    """
    Treina modelo de clustering usando UMAP + HDBSCAN.
    
    Responsabilidades:
    - Redução de dimensionalidade (UMAP)
    - Clustering (HDBSCAN)
    - Busca bayesiana de hiperparâmetros
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_clusters: Optional[hdbscan.HDBSCAN] = None
        self.umap_model: Optional[umap.UMAP] = None
        
    def train(self, X: np.ndarray) -> TrainingResult:
        """
        Executa treino completo com busca de hiperparâmetros.
        
        Args:
            X: Dados normalizados para treino
            
        Returns:
            TrainingResult com métricas e parâmetros
        """
        logger.info("=" * 60)
        logger.info("[CLUSTERING TRAINING] INICIANDO TREINAMENTO")
        logger.info("=" * 60)
        logger.info(f"[CLUSTERING TRAINING] Dados de entrada: {X.shape}")
        logger.info(f"[CLUSTERING TRAINING] Configuração: {self.config}")
        
        try:
            # Busca de hiperparâmetros
            best_params, best_clusters, trials = self._bayesian_search(X)
            
            self.best_params = best_params
            self.best_clusters = best_clusters
            
            # Calcula métricas
            labels = best_clusters.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Silhouette (exclui outliers)
            mask = labels != -1
            if mask.sum() > 1 and n_clusters > 1:
                sil_score = silhouette_score(X[mask], labels[mask])
            else:
                sil_score = 0.0
            
            # Distribuição
            unique, counts = np.unique(labels, return_counts=True)
            distribution = dict(zip(unique.tolist(), counts.tolist()))
            
            # Versão do modelo
            model_version = datetime.now().strftime("v%Y%m%d_%H%M%S")
            
            logger.info("=" * 60)
            logger.info("[CLUSTERING TRAINING] TREINAMENTO CONCLUÍDO COM SUCESSO")
            logger.info(f"[CLUSTERING TRAINING] Versão: {model_version}")
            logger.info(f"[CLUSTERING TRAINING] Clusters encontrados: {n_clusters}")
            logger.info(f"[CLUSTERING TRAINING] Silhouette Score: {sil_score:.4f}")
            logger.info(f"[CLUSTERING TRAINING] Distribuição: {distribution}")
            logger.info(f"[CLUSTERING TRAINING] Melhores parâmetros: {best_params}")
            logger.info("=" * 60)
            
            return TrainingResult(
                success=True,
                model_version=model_version,
                best_params=best_params,
                n_clusters=n_clusters,
                silhouette=sil_score,
                cluster_distribution=distribution
            )
            
        except Exception as e:
            logger.error(f"[CLUSTERING TRAINING] ERRO NO TREINAMENTO: {str(e)}")
            return TrainingResult(
                success=False,
                model_version="",
                best_params={},
                n_clusters=0,
                silhouette=0.0,
                cluster_distribution={},
                error_message=str(e)
            )
    
    def _generate_clusters(
        self,
        X: np.ndarray,
        n_neighbors: int,
        n_components: int,
        min_cluster_size: int,
        min_samples: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> hdbscan.HDBSCAN:
        """
        Gera clusters após redução de dimensionalidade com UMAP.
        
        Args:
            X: Dados normalizados
            n_neighbors: Hiperparâmetro UMAP
            n_components: Dimensões de saída UMAP
            min_cluster_size: Hiperparâmetro HDBSCAN
            min_samples: Hiperparâmetro HDBSCAN
            random_state: Semente para reprodutibilidade
            
        Returns:
            Objeto HDBSCAN com clusters
        """
        # UMAP para redução de dimensionalidade
        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',
            random_state=random_state
        )
        umap_embeddings = self.umap_model.fit_transform(X)
        
        # HDBSCAN para clustering
        clusters = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            gen_min_span_tree=True,
            cluster_selection_method='eom'
        ).fit(umap_embeddings)
        
        return clusters
    
    def _score_clusters(
        self,
        clusters: hdbscan.HDBSCAN,
        prob_threshold: float = 0.05
    ) -> Tuple[int, float]:
        """
        Calcula score do clustering.
        
        Args:
            clusters: Objeto HDBSCAN
            prob_threshold: Limiar de confiança
            
        Returns:
            (número de clusters, custo)
        """
        cluster_labels = clusters.labels_
        label_count = len(np.unique(cluster_labels))
        total_num = len(clusters.labels_)
        cost = np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num
        
        return label_count, cost
    
    def _objective(
        self,
        params: Dict[str, Any],
        X: np.ndarray,
        label_lower: int,
        label_upper: int
    ) -> Dict[str, Any]:
        """
        Função objetivo para otimização bayesiana.
        
        Args:
            params: Parâmetros a avaliar
            X: Dados
            label_lower: Mínimo de clusters desejado
            label_upper: Máximo de clusters desejado
            
        Returns:
            Dict com loss, label_count e status
        """
        clusters = self._generate_clusters(
            X,
            n_neighbors=params['n_neighbors'],
            n_components=params['n_components'],
            min_cluster_size=params['min_cluster_size'],
            min_samples=params.get('min_samples'),
            random_state=params.get('random_state')
        )
        
        label_count, cost = self._score_clusters(clusters, self.config.prob_threshold)
        
        # Penalidade se fora do range desejado
        if (label_count < label_lower) or (label_count > label_upper):
            penalty = 0.15
        else:
            penalty = 0
        
        loss = cost + penalty
        
        return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}
    
    def _bayesian_search(
        self,
        X: np.ndarray
    ) -> Tuple[Dict[str, Any], hdbscan.HDBSCAN, Trials]:
        """
        Busca bayesiana de hiperparâmetros.
        
        Args:
            X: Dados normalizados
            
        Returns:
            (melhores parâmetros, melhor modelo, trials)
        """
        logger.info("[CLUSTERING TRAINING] Iniciando busca bayesiana de hiperparâmetros...")
        
        # Define espaço de busca
        space = {
            "n_neighbors": hp.choice('n_neighbors', range(*self.config.n_neighbors_range)),
            "n_components": hp.choice('n_components', range(*self.config.n_components_range)),
            "min_cluster_size": hp.choice('min_cluster_size', range(*self.config.min_cluster_size_range)),
            "min_samples": hp.choice('min_samples', range(*self.config.min_samples_range)),
            "random_state": self.config.random_state
        }
        
        trials = Trials()
        
        fmin_objective = partial(
            self._objective,
            X=X,
            label_lower=self.config.label_lower,
            label_upper=self.config.label_upper
        )
        
        best = fmin(
            fmin_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.config.max_evals,
            trials=trials,
            verbose=False
        )
        
        best_params = space_eval(space, best)
        
        logger.info(f"[CLUSTERING TRAINING] Busca concluída após {self.config.max_evals} avaliações")
        logger.info(f"[CLUSTERING TRAINING] Melhor label_count: {trials.best_trial['result']['label_count']}")
        
        # Treina modelo final com melhores parâmetros
        best_clusters = self._generate_clusters(
            X,
            n_neighbors=best_params['n_neighbors'],
            n_components=best_params['n_components'],
            min_cluster_size=best_params['min_cluster_size'],
            min_samples=best_params['min_samples'],
            random_state=best_params['random_state']
        )
        
        return best_params, best_clusters, trials


# ============================================================================
# CLASSE: ClusterPredictor
# ============================================================================

class ClusterPredictor:
    """
    Faz inferência usando modelo treinado.
    
    Responsabilidades:
    - Atribuir cluster a novos dados
    - Calcular confiança da predição
    """
    
    def __init__(
        self,
        clusters: Optional[hdbscan.HDBSCAN] = None,
        umap_model: Optional[umap.UMAP] = None
    ):
        self.clusters = clusters
        self.umap_model = umap_model
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediz clusters para novos dados.
        
        Args:
            X: Dados normalizados
            
        Returns:
            (labels, probabilities)
        """
        if self.clusters is None or self.umap_model is None:
            raise PredictionError("Modelo não carregado. Execute o treinamento primeiro.")
        
        logger.info(f"[CLUSTERING PREDICTION] Predizendo {X.shape[0]} amostras...")
        
        # Transforma com UMAP
        X_umap = self.umap_model.transform(X)
        
        # Prediz com HDBSCAN
        labels, probabilities = hdbscan.approximate_predict(self.clusters, X_umap)
        
        logger.info(f"[CLUSTERING PREDICTION] Predição concluída")
        
        return labels, probabilities
    
    def predict_single(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Prediz cluster para uma única amostra.
        
        Args:
            X: Dados normalizados (1, n_features)
            
        Returns:
            (cluster_id, confidence)
        """
        labels, probabilities = self.predict(X)
        return int(labels[0]), float(probabilities[0])


# ============================================================================
# CLASSE: ProfileMapper
# ============================================================================

class ProfileMapper:
    """
    Mapeia clusters numéricos para perfis de negócio.
    
    Responsabilidades:
    - Converter cluster_id → nome do perfil
    - Fornecer descrições dos perfis
    - Fornecer recomendações por perfil
    """
    
    # Mapeamento padrão (pode ser atualizado após análise)
    DEFAULT_CLUSTER_TO_PROFILE = {
        -1: 'Avaliar',
        0: 'Em Desenvolvimento',
        1: 'Em Desenvolvimento',
        2: 'Crítico',
        3: 'Atenção',
        4: 'Crítico',
        5: 'Destaque',
        6: 'Em Desenvolvimento',
        7: 'Destaque',
        8: 'Atenção',
        9: 'Atenção'
    }
    
    PROFILE_DESCRIPTIONS = {
        'Crítico': 'Alunos com aprendizado muito baixo ou nulo, requerem intervenção urgente. '
                   'Maioria na Pedra Quartzo com IAA próximo de zero.',
        'Atenção': 'Alunos estagnados há muito tempo no programa ou com dificuldade acadêmica '
                   'apesar de engajamento. Requerem acompanhamento próximo.',
        'Em Desenvolvimento': 'Alunos no caminho certo, ajustados à sua fase. '
                               'Bom desempenho mas ainda não atingiram Ponto de Virada.',
        'Destaque': 'Alunos de alto desempenho com engajamento exemplar. '
                    'Podem ser mentores para outros alunos.',
        'Avaliar': 'Perfil atípico que não se encaixa em nenhum padrão claro. '
                   'Requer avaliação individual.'
    }
    
    PROFILE_RECOMMENDATIONS = {
        'Crítico': [
            'Avaliação psicopedagógica individual urgente',
            'Reforço intensivo em alfabetização/matemática básica',
            'Acompanhamento psicológico prioritário',
            'Contato com família para entender contexto'
        ],
        'Atenção': [
            'Identificar se o problema é acadêmico ou emocional',
            'Tutoria em pequenos grupos',
            'Monitorar evolução a cada 2 meses',
            'Revisar metodologia de ensino para o grupo'
        ],
        'Em Desenvolvimento': [
            'Manter acompanhamento regular',
            'Incentivar participação em atividades extras',
            'Trabalhar meta de atingir Ponto de Virada',
            'Reconhecer progressos para manter motivação'
        ],
        'Destaque': [
            'Programa de mentoria para ajudar outros alunos',
            'Desafios acadêmicos adicionais',
            'Preparação para oportunidades de bolsa',
            'Reconhecimento público das conquistas'
        ],
        'Avaliar': [
            'Análise individual do histórico completo',
            'Entrevista com professores que acompanham',
            'Avaliação psicológica se necessário',
            'Definir plano personalizado após análise'
        ]
    }
    
    def __init__(self, custom_mapping: Optional[Dict[int, str]] = None):
        self.cluster_to_profile = custom_mapping or self.DEFAULT_CLUSTER_TO_PROFILE.copy()
        
    def get_profile(self, cluster_id: int) -> str:
        """Retorna nome do perfil para um cluster."""
        return self.cluster_to_profile.get(cluster_id, 'Desconhecido')
    
    def get_description(self, profile: str) -> str:
        """Retorna descrição do perfil."""
        return self.PROFILE_DESCRIPTIONS.get(profile, 'Perfil não reconhecido.')
    
    def get_recommendations(self, profile: str) -> List[str]:
        """Retorna lista de recomendações para o perfil."""
        return self.PROFILE_RECOMMENDATIONS.get(profile, ['Avaliar caso individualmente.'])
    
    def update_mapping(self, new_mapping: Dict[int, str]):
        """Atualiza o mapeamento cluster → perfil."""
        self.cluster_to_profile.update(new_mapping)
        logger.info(f"[PROFILE MAPPER] Mapeamento atualizado: {self.cluster_to_profile}")
    
    def get_full_profile(self, cluster_id: int) -> Dict[str, Any]:
        """Retorna informações completas do perfil."""
        profile = self.get_profile(cluster_id)
        return {
            'cluster_id': cluster_id,
            'profile': profile,
            'description': self.get_description(profile),
            'recommendations': self.get_recommendations(profile)
        }


# ============================================================================
# CLASSE: ClusterEvaluator
# ============================================================================

class ClusterEvaluator:
    """
    Avalia qualidade do clustering e detecta drift.
    
    Responsabilidades:
    - Calcular Silhouette Score
    - Calcular distribuição dos clusters
    - Detectar drift nos dados
    """
    
    PSI_THRESHOLD_WARNING = 0.1
    PSI_THRESHOLD_CRITICAL = 0.2
    
    def evaluate(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """
        Avalia clustering completo.
        
        Args:
            X: Dados normalizados
            labels: Labels dos clusters
            probabilities: Probabilidades de atribuição
            
        Returns:
            EvaluationMetrics
        """
        logger.info("[CLUSTERING EVALUATION] Calculando métricas...")
        
        # Número de clusters (excluindo outliers)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Número de outliers
        n_outliers = (labels == -1).sum()
        
        # Silhouette (exclui outliers)
        mask = labels != -1
        if mask.sum() > 1 and n_clusters > 1:
            sil = silhouette_score(X[mask], labels[mask])
        else:
            sil = 0.0
        
        # Distribuição
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        # Ratio de baixa confiança
        if probabilities is not None:
            low_conf_ratio = (probabilities < 0.5).sum() / len(probabilities)
        else:
            low_conf_ratio = 0.0
        
        metrics = EvaluationMetrics(
            silhouette=sil,
            n_clusters=n_clusters,
            n_outliers=n_outliers,
            cluster_distribution=distribution,
            low_confidence_ratio=low_conf_ratio
        )
        
        logger.info(f"[CLUSTERING EVALUATION] Silhouette: {sil:.4f}")
        logger.info(f"[CLUSTERING EVALUATION] Clusters: {n_clusters}, Outliers: {n_outliers}")
        
        return metrics
    
    def calculate_psi(
        self,
        train_data: np.ndarray,
        prod_data: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calcula Population Stability Index.
        
        PSI < 0.1: Sem mudança significativa
        0.1 <= PSI < 0.2: Mudança moderada
        PSI >= 0.2: Mudança significativa
        
        Args:
            train_data: Dados de treino
            prod_data: Dados de produção
            bins: Número de bins
            
        Returns:
            PSI score
        """
        # Cria bins baseados nos dados de treino
        breakpoints = np.percentile(train_data, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        # Calcula proporções em cada bin
        train_counts, _ = np.histogram(train_data, bins=breakpoints)
        prod_counts, _ = np.histogram(prod_data, bins=breakpoints)
        
        # Normaliza
        train_pct = train_counts / len(train_data)
        prod_pct = prod_counts / len(prod_data)
        
        # Evita divisão por zero
        train_pct = np.clip(train_pct, 0.0001, None)
        prod_pct = np.clip(prod_pct, 0.0001, None)
        
        # Calcula PSI
        psi = np.sum((prod_pct - train_pct) * np.log(prod_pct / train_pct))
        
        return float(psi)
    
    def detect_drift(
        self,
        X_train: np.ndarray,
        X_new: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> DriftResult:
        """
        Detecta drift entre dados de treino e novos dados.
        
        Args:
            X_train: Dados de treino
            X_new: Novos dados
            feature_names: Nomes das features
            
        Returns:
            DriftResult
        """
        logger.info("[DRIFT DETECTION] Analisando drift nos dados...")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        psi_scores = {}
        features_with_drift = []
        
        for i, name in enumerate(feature_names):
            psi = self.calculate_psi(X_train[:, i], X_new[:, i])
            psi_scores[name] = round(psi, 4)
            
            if psi >= self.PSI_THRESHOLD_CRITICAL:
                features_with_drift.append(name)
                logger.warning(f"[DRIFT DETECTION] DRIFT CRÍTICO em {name}: PSI={psi:.4f}")
            elif psi >= self.PSI_THRESHOLD_WARNING:
                logger.warning(f"[DRIFT DETECTION] Drift moderado em {name}: PSI={psi:.4f}")
        
        drift_detected = len(features_with_drift) > 0
        
        if drift_detected:
            recommendation = "Retreinamento recomendado devido a drift significativo."
        else:
            recommendation = "Dados estáveis, retreinamento não necessário."
        
        logger.info(f"[DRIFT DETECTION] Drift detectado: {drift_detected}")
        
        return DriftResult(
            drift_detected=drift_detected,
            psi_scores=psi_scores,
            features_with_drift=features_with_drift,
            recommendation=recommendation
        )