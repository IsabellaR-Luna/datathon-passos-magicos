# monitoring.py
"""
Rotas de monitoramento e detecção de drift.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.drift_service import get_drift_service, DriftService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoramento"])



class FeatureDriftSchema(BaseModel):
    """Schema para drift de uma feature."""
    feature: str
    reference_mean: float
    reference_std: float
    current_mean: float
    current_std: float
    mean_diff_pct: float
    psi: float
    ks_statistic: float
    ks_pvalue: float
    has_drift: bool
    severity: str


class DriftReportSchema(BaseModel):
    """Schema para relatório de drift."""
    reference_period: str
    comparison_period: str
    reference_samples: int
    comparison_samples: int
    features_analyzed: int
    features_with_drift: int
    overall_drift: bool
    recommendation: str
    feature_details: List[FeatureDriftSchema]


class FeatureStatsSchema(BaseModel):
    """Schema para estatísticas de uma feature."""
    mean: float
    std: float
    min: float
    max: float
    median: float


class PeriodStatsSchema(BaseModel):
    """Schema para estatísticas de um período."""
    period: str
    samples: int
    features: dict


class DriftSummarySchema(BaseModel):
    """Schema para resumo de drift."""
    status: str
    reference_period: str
    comparison_periods: List[str]
    overall_drift: bool
    features_monitored: int
    alerts: List[str]


def get_service() -> DriftService:
    """Obtém instância do serviço de drift."""
    try:
        return get_drift_service()
    except FileNotFoundError as e:
        logger.error(f"[MONITORING] Arquivo não encontrado: {e}")
        raise HTTPException(
            status_code=503,
            detail="Dados de referência não disponíveis. Verifique se o arquivo Excel existe."
        )
    except Exception as e:
        logger.error(f"[MONITORING] Erro ao inicializar serviço: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao inicializar serviço de monitoramento: {str(e)}"
        )



@router.get("/drift", response_model=DriftReportSchema)
async def analyze_drift(
    comparison: Optional[str] = Query(
        None,
        description="Período para comparar (PEDE2023 ou PEDE2024). Default: PEDE2024"
    )
):
    """
    Analisa drift entre dados de treino e dados de comparação.
    
    Compara a distribuição das features do modelo de treino (PEDE2022)
    com dados mais recentes para detectar mudanças significativas.
    
    **Métricas utilizadas:**
    - **PSI (Population Stability Index)**: Mede mudança na distribuição
      - < 0.1: Sem mudança
      - 0.1-0.2: Mudança moderada
      - > 0.2: Mudança significativa
    - **KS Test**: Teste estatístico de diferença de distribuições
    - **Diferença de média**: Variação percentual na média
    
    Returns:
        Relatório detalhado de drift por feature
    """
    logger.info(f"[MONITORING] GET /monitoring/drift - comparison={comparison}")
    
    service = get_service()
    
    try:
        report = service.analyze(comparison)
        return DriftReportSchema(**report.to_dict())
    except Exception as e:
        logger.error(f"[MONITORING] Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/all", response_model=List[DriftReportSchema])
async def analyze_all_periods():
    """
    Analisa drift para todos os períodos disponíveis.
    
    Compara PEDE2022 (treino) com PEDE2023 e PEDE2024.
    
    Returns:
        Lista de relatórios de drift
    """
    logger.info("[MONITORING] GET /monitoring/drift/all")
    
    service = get_service()
    
    try:
        reports = service.analyze_all_periods()
        return [DriftReportSchema(**r.to_dict()) for r in reports]
    except Exception as e:
        logger.error(f"[MONITORING] Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/summary", response_model=DriftSummarySchema)
async def get_drift_summary():
    """
    Retorna resumo do status de drift.
    
    Útil para dashboards e alertas rápidos.
    
    Returns:
        Resumo com status geral e alertas
    """
    logger.info("[MONITORING] GET /monitoring/drift/summary")
    
    service = get_service()
    
    try:
        # Analisa o período mais recente
        report = service.analyze()
        
        # Gera alertas
        alerts = []
        for feature in report.feature_details:
            if feature.has_drift:
                if feature.severity == "high":
                    alerts.append(f"🔴 {feature.feature}: drift alto (PSI={feature.psi:.3f})")
                elif feature.severity == "medium":
                    alerts.append(f"🟡 {feature.feature}: drift moderado (PSI={feature.psi:.3f})")
        
        return DriftSummarySchema(
            status="warning" if report.overall_drift else "ok",
            reference_period=report.reference_period,
            comparison_periods=service.config.comparison_sheets,
            overall_drift=report.overall_drift,
            features_monitored=report.features_analyzed,
            alerts=alerts
        )
    except Exception as e:
        logger.error(f"[MONITORING] Erro no resumo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/reference", response_model=PeriodStatsSchema)
async def get_reference_stats():
    """
    Retorna estatísticas dos dados de referência (treino).
    
    Returns:
        Estatísticas descritivas por feature
    """
    logger.info("[MONITORING] GET /monitoring/stats/reference")
    
    service = get_service()
    
    try:
        stats = service.get_reference_stats()
        return PeriodStatsSchema(**stats)
    except Exception as e:
        logger.error(f"[MONITORING] Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{period}", response_model=PeriodStatsSchema)
async def get_period_stats(period: str):
    """
    Retorna estatísticas de um período específico.
    
    Args:
        period: PEDE2022, PEDE2023 ou PEDE2024
    
    Returns:
        Estatísticas descritivas por feature
    """
    logger.info(f"[MONITORING] GET /monitoring/stats/{period}")
    
    valid_periods = ["PEDE2022", "PEDE2023", "PEDE2024"]
    if period not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Período inválido. Use: {valid_periods}"
        )
    
    service = get_service()
    
    try:
        if period == "PEDE2022":
            stats = service.get_reference_stats()
        else:
            stats = service.get_comparison_stats(period)
        
        return PeriodStatsSchema(**stats)
    except Exception as e:
        logger.error(f"[MONITORING] Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))