# cluster.py
"""
Endpoints de clustering da API.

Endpoints:
    - GET /clusters/summary - Estatísticas dos perfis
    - GET /clusters/students - Lista alunos com filtros
    - GET /clusters/student/{ra} - Info de um aluno
    - GET /clusters/profiles - Lista perfis disponíveis
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from app.services.chat.core import QueryExecutor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clusters", tags=["Clusters"])

# Executor de queries (será injetado via dependency)
query_executor: Optional[QueryExecutor] = None


def get_executor() -> QueryExecutor:
    """Retorna o executor de queries."""
    if query_executor is None:
        raise HTTPException(status_code=500, detail="Query executor não configurado")
    return query_executor


# ============================================================================
# SCHEMAS
# ============================================================================

class ProfileStats(BaseModel):
    """Estatísticas de um perfil."""
    perfil: str
    total: int
    media_iaa: float
    media_ieg: float
    media_ida: float


class ClusterSummary(BaseModel):
    """Resumo geral dos clusters."""
    total_alunos: int
    distribuicao: dict
    estatisticas_por_perfil: List[ProfileStats]


class StudentBasic(BaseModel):
    """Dados básicos de um aluno."""
    ra: str
    nome: str
    perfil: str
    cluster_id: int
    iaa: float
    ieg: float
    ida: float
    defasagem: int


class StudentDetail(BaseModel):
    """Dados completos de um aluno."""
    ra: str
    nome: str
    idade: Optional[int]
    genero: Optional[str]
    turma: Optional[str]
    fase: Optional[int]
    fase_ideal: Optional[str]
    instituicao_ensino: Optional[str]
    ano_ingresso: Optional[int]
    
    # Indicadores
    iaa: Optional[float]
    ieg: Optional[float]
    ips: Optional[float]
    ida: Optional[float]
    ipv: Optional[float]
    ian: Optional[float]
    inde: Optional[float]
    
    # Notas
    nota_matematica: Optional[float]
    nota_portugues: Optional[float]
    nota_ingles: Optional[float]
    
    # Status
    defasagem: Optional[int]
    atingiu_ponto_virada: Optional[str]
    pedra_2022: Optional[str]
    indicado_bolsa: Optional[str]
    
    # Cluster
    cluster_id: int
    perfil: str
    
    # Recomendações
    recomendacoes: List[str] = Field(default_factory=list)


class ProfileInfo(BaseModel):
    """Informações de um perfil."""
    nome: str
    descricao: str
    recomendacoes: List[str]
    total_alunos: int


class StudentsResponse(BaseModel):
    """Resposta da lista de alunos."""
    total: int
    alunos: List[StudentBasic]


# ============================================================================
# MAPEAMENTO DE PERFIS (para recomendações)
# ============================================================================

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

PROFILE_DESCRIPTIONS = {
    'Crítico': 'Alunos com aprendizado muito baixo ou nulo, requerem intervenção urgente.',
    'Atenção': 'Alunos estagnados há muito tempo ou com dificuldade acadêmica apesar de engajamento.',
    'Em Desenvolvimento': 'Alunos no caminho certo, ajustados à sua fase. Bom desempenho mas ainda não atingiram Ponto de Virada.',
    'Destaque': 'Alunos de alto desempenho com engajamento exemplar.',
    'Avaliar': 'Perfil atípico que não se encaixa em nenhum padrão claro.'
}


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/summary", response_model=ClusterSummary)
async def get_cluster_summary():
    """
    Retorna estatísticas agregadas dos clusters.
    
    Returns:
        Resumo com total de alunos, distribuição e médias por perfil
    """
    logger.info("[CLUSTER API] GET /clusters/summary")
    
    executor = get_executor()
    
    # Total de alunos
    result_total = executor.execute("SELECT COUNT(*) as total FROM alunos")
    if not result_total.success:
        raise HTTPException(status_code=500, detail="Erro ao consultar banco")
    
    total_alunos = result_total.data[0]['total']
    
    # Distribuição por perfil
    result_dist = executor.execute(
        "SELECT perfil, COUNT(*) as total FROM alunos GROUP BY perfil ORDER BY total DESC"
    )
    if not result_dist.success:
        raise HTTPException(status_code=500, detail="Erro ao consultar banco")
    
    distribuicao = {row['perfil']: row['total'] for row in result_dist.data}
    
    # Estatísticas por perfil
    result_stats = executor.execute("""
        SELECT 
            perfil,
            COUNT(*) as total,
            ROUND(AVG(iaa), 2) as media_iaa,
            ROUND(AVG(ieg), 2) as media_ieg,
            ROUND(AVG(ida), 2) as media_ida
        FROM alunos 
        GROUP BY perfil 
        ORDER BY total DESC
    """)
    if not result_stats.success:
        raise HTTPException(status_code=500, detail="Erro ao consultar banco")
    
    estatisticas = [
        ProfileStats(
            perfil=row['perfil'],
            total=row['total'],
            media_iaa=row['media_iaa'] or 0,
            media_ieg=row['media_ieg'] or 0,
            media_ida=row['media_ida'] or 0
        )
        for row in result_stats.data
    ]
    
    return ClusterSummary(
        total_alunos=total_alunos,
        distribuicao=distribuicao,
        estatisticas_por_perfil=estatisticas
    )


@router.get("/students", response_model=StudentsResponse)
async def get_students(
    perfil: Optional[str] = Query(None, description="Filtrar por perfil"),
    turma: Optional[str] = Query(None, description="Filtrar por turma"),
    min_iaa: Optional[float] = Query(None, description="IAA mínimo"),
    max_iaa: Optional[float] = Query(None, description="IAA máximo"),
    limit: int = Query(20, ge=1, le=100, description="Limite de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginação")
):
    """
    Lista alunos com filtros opcionais.
    
    Args:
        perfil: Filtrar por perfil (Crítico, Atenção, etc.)
        turma: Filtrar por turma (A, B, C, etc.)
        min_iaa: IAA mínimo
        max_iaa: IAA máximo
        limit: Limite de resultados (máx 100)
        offset: Offset para paginação
    
    Returns:
        Lista de alunos com dados básicos
    """
    logger.info(f"[CLUSTER API] GET /clusters/students - perfil={perfil}, turma={turma}")
    
    executor = get_executor()
    
    # Monta query com filtros
    conditions = []
    if perfil:
        conditions.append(f"perfil = '{perfil}'")
    if turma:
        conditions.append(f"turma = '{turma}'")
    if min_iaa is not None:
        conditions.append(f"iaa >= {min_iaa}")
    if max_iaa is not None:
        conditions.append(f"iaa <= {max_iaa}")
    
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    
    # Query principal
    query = f"""
        SELECT ra, nome, perfil, cluster_id, iaa, ieg, ida, defasagem
        FROM alunos
        {where_clause}
        ORDER BY perfil, nome
        LIMIT {limit} OFFSET {offset}
    """
    
    result = executor.execute(query)
    if not result.success:
        raise HTTPException(status_code=500, detail="Erro ao consultar banco")
    
    # Conta total
    count_query = f"SELECT COUNT(*) as total FROM alunos {where_clause}"
    result_count = executor.execute(count_query)
    total = result_count.data[0]['total'] if result_count.success else 0
    
    alunos = [
        StudentBasic(
            ra=row['ra'],
            nome=row['nome'],
            perfil=row['perfil'],
            cluster_id=row['cluster_id'],
            iaa=row['iaa'] or 0,
            ieg=row['ieg'] or 0,
            ida=row['ida'] or 0,
            defasagem=row['defasagem'] or 0
        )
        for row in result.data
    ]
    
    return StudentsResponse(total=total, alunos=alunos)


@router.get("/student/{ra}", response_model=StudentDetail)
async def get_student(ra: str):
    """
    Retorna dados completos de um aluno.
    
    Args:
        ra: Registro do aluno (ex: RA-123)
    
    Returns:
        Dados completos do aluno com recomendações
    """
    logger.info(f"[CLUSTER API] GET /clusters/student/{ra}")
    
    executor = get_executor()
    
    result = executor.execute(f"SELECT * FROM alunos WHERE ra = '{ra}'")
    
    if not result.success:
        raise HTTPException(status_code=500, detail="Erro ao consultar banco")
    
    if result.row_count == 0:
        raise HTTPException(status_code=404, detail=f"Aluno {ra} não encontrado")
    
    row = result.data[0]
    perfil = row.get('perfil', 'Avaliar')
    
    return StudentDetail(
        ra=row['ra'],
        nome=row['nome'],
        idade=row.get('idade'),
        genero=row.get('genero'),
        turma=row.get('turma'),
        fase=row.get('fase'),
        fase_ideal=row.get('fase_ideal'),
        instituicao_ensino=row.get('instituicao_ensino'),
        ano_ingresso=row.get('ano_ingresso'),
        iaa=row.get('iaa'),
        ieg=row.get('ieg'),
        ips=row.get('ips'),
        ida=row.get('ida'),
        ipv=row.get('ipv'),
        ian=row.get('ian'),
        inde=row.get('inde'),
        nota_matematica=row.get('nota_matematica'),
        nota_portugues=row.get('nota_portugues'),
        nota_ingles=row.get('nota_ingles'),
        defasagem=row.get('defasagem'),
        atingiu_ponto_virada=row.get('atingiu_ponto_virada'),
        pedra_2022=row.get('pedra_2022'),
        indicado_bolsa=row.get('indicado_bolsa'),
        cluster_id=row.get('cluster_id', -1),
        perfil=perfil,
        recomendacoes=PROFILE_RECOMMENDATIONS.get(perfil, [])
    )


@router.get("/profiles", response_model=List[ProfileInfo])
async def get_profiles():
    """
    Lista todos os perfis disponíveis com descrições.
    
    Returns:
        Lista de perfis com descrição e recomendações
    """
    logger.info("[CLUSTER API] GET /clusters/profiles")
    
    executor = get_executor()
    
    # Conta alunos por perfil
    result = executor.execute(
        "SELECT perfil, COUNT(*) as total FROM alunos GROUP BY perfil"
    )
    
    counts = {}
    if result.success:
        counts = {row['perfil']: row['total'] for row in result.data}
    
    profiles = []
    for nome in ['Crítico', 'Atenção', 'Em Desenvolvimento', 'Destaque', 'Avaliar']:
        profiles.append(ProfileInfo(
            nome=nome,
            descricao=PROFILE_DESCRIPTIONS.get(nome, ''),
            recomendacoes=PROFILE_RECOMMENDATIONS.get(nome, []),
            total_alunos=counts.get(nome, 0)
        ))
    
    return profiles