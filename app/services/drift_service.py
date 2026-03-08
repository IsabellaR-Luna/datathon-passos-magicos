# drift_service.py
"""
Serviço de detecção de Data Drift.

Compara a distribuição dos dados de treino (PEDE 2022) com dados
de produção (PEDE 2023/2024) para detectar mudanças significativas.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuração do serviço de drift."""
    
    excel_path: str = "data/BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx"
    reference_sheet: str = "PEDE2022"  # Dados de treino
    comparison_sheets: List[str] = None  # Dados de "produção"
    
    # Features para monitorar
    features: List[str] = None
    
    # Thresholds para alertas
    psi_threshold: float = 0.2  # Population Stability Index
    ks_pvalue_threshold: float = 0.05  # Kolmogorov-Smirnov test
    mean_diff_threshold: float = 0.15  # 15% de diferença na média
    
    def __post_init__(self):
        if self.comparison_sheets is None:
            self.comparison_sheets = ["PEDE2023", "PEDE2024"]
        
        if self.features is None:
            self.features = ["IAA", "IEG", "IPS", "IDA", "IPV", "IAN"]


@dataclass
class FeatureDrift:
    """Resultado de drift para uma feature."""
    
    feature: str
    reference_mean: float
    reference_std: float
    current_mean: float
    current_std: float
    mean_diff_pct: float  # Diferença percentual na média
    psi: float  # Population Stability Index
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_pvalue: float  # Kolmogorov-Smirnov p-value
    has_drift: bool  # Se drift foi detectado
    severity: str  # "low", "medium", "high"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "reference_mean": round(self.reference_mean, 3),
            "reference_std": round(self.reference_std, 3),
            "current_mean": round(self.current_mean, 3),
            "current_std": round(self.current_std, 3),
            "mean_diff_pct": round(self.mean_diff_pct, 2),
            "psi": round(self.psi, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_pvalue": round(self.ks_pvalue, 4),
            "has_drift": self.has_drift,
            "severity": self.severity
        }


@dataclass
class DriftReport:
    """Relatório completo de drift."""
    
    reference_period: str
    comparison_period: str
    reference_samples: int
    comparison_samples: int
    features_analyzed: int
    features_with_drift: int
    overall_drift: bool
    recommendation: str
    feature_details: List[FeatureDrift]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_period": self.reference_period,
            "comparison_period": self.comparison_period,
            "reference_samples": self.reference_samples,
            "comparison_samples": self.comparison_samples,
            "features_analyzed": self.features_analyzed,
            "features_with_drift": self.features_with_drift,
            "overall_drift": self.overall_drift,
            "recommendation": self.recommendation,
            "feature_details": [f.to_dict() for f in self.feature_details]
        }



class DriftService:
    """
    Serviço de detecção de Data Drift.
    
    Compara distribuições estatísticas entre dados de referência (treino)
    e dados de comparação (produção) para detectar mudanças significativas.
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        """
        Inicializa o serviço de drift.
        
        Args:
            config: Configuração do serviço
        """
        self.config = config or DriftConfig()
        self._reference_data: Optional[pd.DataFrame] = None
        self._reference_stats: Optional[Dict] = None
        
        logger.info("=" * 60)
        logger.info("[DRIFT SERVICE] SERVIÇO INICIALIZADO")
        logger.info(f"[DRIFT SERVICE] Referência: {self.config.reference_sheet}")
        logger.info(f"[DRIFT SERVICE] Features: {self.config.features}")
        logger.info("=" * 60)
    
    def _load_data(self, sheet_name: str) -> pd.DataFrame:
        """Carrega dados de uma aba do Excel."""
        try:
            df = pd.read_excel(
                self.config.excel_path,
                sheet_name=sheet_name
            )
            
            # Filtra apenas as features que existem
            available_features = [f for f in self.config.features if f in df.columns]
            
            return df[available_features].dropna()
            
        except Exception as e:
            logger.error(f"[DRIFT SERVICE] Erro ao carregar {sheet_name}: {e}")
            raise
    
    def _get_reference_data(self) -> pd.DataFrame:
        """Retorna dados de referência (com cache)."""
        if self._reference_data is None:
            self._reference_data = self._load_data(self.config.reference_sheet)
            logger.info(f"[DRIFT SERVICE] Dados de referência carregados: {len(self._reference_data)} amostras")
        
        return self._reference_data
    
    def _get_reference_stats(self) -> Dict[str, Dict[str, float]]:
        """Retorna estatísticas de referência (com cache)."""
        if self._reference_stats is None:
            df = self._get_reference_data()
            
            self._reference_stats = {}
            for feature in self.config.features:
                if feature in df.columns:
                    self._reference_stats[feature] = {
                        "mean": df[feature].mean(),
                        "std": df[feature].std(),
                        "min": df[feature].min(),
                        "max": df[feature].max(),
                        "median": df[feature].median(),
                        "q25": df[feature].quantile(0.25),
                        "q75": df[feature].quantile(0.75)
                    }
        
        return self._reference_stats
    
    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calcula o Population Stability Index (PSI).
        
        PSI < 0.1: Sem mudança significativa
        PSI 0.1-0.2: Mudança moderada
        PSI > 0.2: Mudança significativa
        """
        # Define bins baseado nos dados de referência
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Calcula proporções em cada bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Normaliza para proporções
        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)
        
        # Evita divisão por zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
        
        # Calcula PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)
    
    def _analyze_feature(
        self,
        feature: str,
        reference_data: pd.Series,
        current_data: pd.Series
    ) -> FeatureDrift:
        """Analisa drift para uma feature específica."""
        
        ref_mean = reference_data.mean()
        ref_std = reference_data.std()
        cur_mean = current_data.mean()
        cur_std = current_data.std()
        
        # Diferença percentual na média
        if ref_mean != 0:
            mean_diff_pct = abs(cur_mean - ref_mean) / abs(ref_mean) * 100
        else:
            mean_diff_pct = 0 if cur_mean == 0 else 100
        
        # PSI
        psi = self._calculate_psi(
            reference_data.values,
            current_data.values
        )
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(
            reference_data.values,
            current_data.values
        )
        
        # Determina se há drift
        has_drift = (
            psi > self.config.psi_threshold or
            ks_pvalue < self.config.ks_pvalue_threshold or
            mean_diff_pct > self.config.mean_diff_threshold * 100
        )
        
        # Determina severidade
        if psi > 0.25 or mean_diff_pct > 25:
            severity = "high"
        elif psi > 0.1 or mean_diff_pct > 10:
            severity = "medium"
        else:
            severity = "low"
        
        return FeatureDrift(
            feature=feature,
            reference_mean=ref_mean,
            reference_std=ref_std,
            current_mean=cur_mean,
            current_std=cur_std,
            mean_diff_pct=mean_diff_pct,
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            has_drift=has_drift,
            severity=severity
        )
    
    def analyze(self, comparison_sheet: str = None) -> DriftReport:
        """
        Analisa drift entre dados de referência e comparação.
        
        Args:
            comparison_sheet: Aba do Excel para comparar (default: mais recente)
        
        Returns:
            Relatório de drift
        """
        if comparison_sheet is None:
            comparison_sheet = self.config.comparison_sheets[-1]  # Mais recente
        
        logger.info(f"[DRIFT SERVICE] Analisando drift: {self.config.reference_sheet} vs {comparison_sheet}")
        
        # Carrega dados
        reference_df = self._get_reference_data()
        comparison_df = self._load_data(comparison_sheet)
        
        # Analisa cada feature
        feature_results = []
        features_with_drift = 0
        
        for feature in self.config.features:
            if feature in reference_df.columns and feature in comparison_df.columns:
                result = self._analyze_feature(
                    feature,
                    reference_df[feature],
                    comparison_df[feature]
                )
                feature_results.append(result)
                
                if result.has_drift:
                    features_with_drift += 1
                    logger.warning(
                        f"[DRIFT SERVICE] Drift detectado em {feature}: "
                        f"PSI={result.psi:.3f}, diff={result.mean_diff_pct:.1f}%"
                    )
        
        # Determina drift geral
        overall_drift = features_with_drift >= len(feature_results) / 2
        
        # Gera recomendação
        if overall_drift:
            recommendation = (
                "⚠️ DRIFT SIGNIFICATIVO DETECTADO. "
                "Recomenda-se retreinar o modelo com dados mais recentes."
            )
        elif features_with_drift > 0:
            recommendation = (
                "⚡ Drift moderado detectado em algumas features. "
                "Monitore a performance do modelo e considere retreinamento."
            )
        else:
            recommendation = (
                "✅ Sem drift significativo. "
                "O modelo permanece válido para os dados atuais."
            )
        
        report = DriftReport(
            reference_period=self.config.reference_sheet,
            comparison_period=comparison_sheet,
            reference_samples=len(reference_df),
            comparison_samples=len(comparison_df),
            features_analyzed=len(feature_results),
            features_with_drift=features_with_drift,
            overall_drift=overall_drift,
            recommendation=recommendation,
            feature_details=feature_results
        )
        
        logger.info(f"[DRIFT SERVICE] Análise concluída: {features_with_drift}/{len(feature_results)} features com drift")
        
        return report
    
    def analyze_all_periods(self) -> List[DriftReport]:
        """
        Analisa drift para todos os períodos de comparação.
        
        Returns:
            Lista de relatórios de drift
        """
        reports = []
        
        for sheet in self.config.comparison_sheets:
            try:
                report = self.analyze(sheet)
                reports.append(report)
            except Exception as e:
                logger.error(f"[DRIFT SERVICE] Erro ao analisar {sheet}: {e}")
        
        return reports
    
    def get_reference_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas dos dados de referência.
        
        Returns:
            Dict com estatísticas por feature
        """
        return {
            "period": self.config.reference_sheet,
            "samples": len(self._get_reference_data()),
            "features": self._get_reference_stats()
        }
    
    def get_comparison_stats(self, sheet: str = None) -> Dict[str, Any]:
        """
        Retorna estatísticas de um período de comparação.
        
        Args:
            sheet: Aba do Excel (default: mais recente)
        
        Returns:
            Dict com estatísticas por feature
        """
        if sheet is None:
            sheet = self.config.comparison_sheets[-1]
        
        df = self._load_data(sheet)
        
        stats = {}
        for feature in self.config.features:
            if feature in df.columns:
                stats[feature] = {
                    "mean": float(df[feature].mean()),
                    "std": float(df[feature].std()),
                    "min": float(df[feature].min()),
                    "max": float(df[feature].max()),
                    "median": float(df[feature].median())
                }
        
        return {
            "period": sheet,
            "samples": len(df),
            "features": stats
        }


# Será inicializado quando a rota for carregada
drift_service: Optional[DriftService] = None


def get_drift_service() -> DriftService:
    """Retorna instância do serviço de drift."""
    global drift_service
    
    if drift_service is None:
        # Tenta encontrar o arquivo Excel
        possible_paths = [
            "data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx",
            "/app/data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx",
        ]
    
        excel_path = None
        for path in possible_paths:
            if Path(path).exists():
                excel_path = path
                break
        
        if excel_path is None:
            raise FileNotFoundError("Arquivo Excel não encontrado")
        
        config = DriftConfig(excel_path=excel_path)
        drift_service = DriftService(config)
    
    return drift_service