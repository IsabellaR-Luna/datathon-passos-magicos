# database_setup.py
"""
Script para criar e popular o banco de dados SQLite com dados dos alunos.

Uso:
    python database_setup.py --input data/alunos_clusterizados.xlsx --output data/passos_magicos.db
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CLUSTER_TO_PROFILE = {
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

# SCHEMA DO BANCO

CREATE_ALUNOS_TABLE = """
CREATE TABLE IF NOT EXISTS alunos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ra TEXT UNIQUE NOT NULL,
    nome TEXT NOT NULL,
    
    -- Dados demográficos
    idade INTEGER,
    genero TEXT,
    ano_nascimento INTEGER,
    ano_ingresso INTEGER,
    instituicao_ensino TEXT,
    
    -- Fase e turma
    fase INTEGER,
    turma TEXT,
    fase_ideal TEXT,
    
    -- Indicadores principais
    iaa REAL,
    ieg REAL,
    ips REAL,
    ida REAL,
    ipv REAL,
    ian REAL,
    inde REAL,
    
    -- Notas por disciplina
    nota_matematica REAL,
    nota_portugues REAL,
    nota_ingles REAL,
    
    -- Defasagem e ponto de virada
    defasagem INTEGER,
    atingiu_ponto_virada TEXT,
    
    -- Pedras (níveis)
    pedra_2020 TEXT,
    pedra_2021 TEXT,
    pedra_2022 TEXT,
    
    -- Avaliações e recomendações
    indicado_bolsa TEXT,
    rec_psicologia TEXT,
    destaque_ieg TEXT,
    destaque_ida TEXT,
    destaque_ipv TEXT,
    
    -- Clustering
    cluster_id INTEGER,
    perfil TEXT,
    
    -- Metadados
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDICES = """
CREATE INDEX IF NOT EXISTS idx_alunos_perfil ON alunos(perfil);
CREATE INDEX IF NOT EXISTS idx_alunos_turma ON alunos(turma);
CREATE INDEX IF NOT EXISTS idx_alunos_fase ON alunos(fase);
CREATE INDEX IF NOT EXISTS idx_alunos_cluster ON alunos(cluster_id);
CREATE INDEX IF NOT EXISTS idx_alunos_defasagem ON alunos(defasagem);
"""

CREATE_PERFIS_TABLE = """
CREATE TABLE IF NOT EXISTS perfis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT UNIQUE NOT NULL,
    descricao TEXT,
    recomendacoes TEXT
);
"""

INSERT_PERFIS = """
INSERT OR REPLACE INTO perfis (nome, descricao, recomendacoes) VALUES
('Crítico', 'Alunos com aprendizado muito baixo ou nulo, requerem intervenção urgente.', 
 'Avaliação psicopedagógica individual urgente|Reforço intensivo em alfabetização/matemática básica|Acompanhamento psicológico prioritário|Contato com família para entender contexto'),

('Atenção', 'Alunos estagnados há muito tempo no programa ou com dificuldade acadêmica apesar de engajamento.',
 'Identificar se o problema é acadêmico ou emocional|Tutoria em pequenos grupos|Monitorar evolução a cada 2 meses|Revisar metodologia de ensino para o grupo'),

('Em Desenvolvimento', 'Alunos no caminho certo, ajustados à sua fase. Bom desempenho mas ainda não atingiram Ponto de Virada.',
 'Manter acompanhamento regular|Incentivar participação em atividades extras|Trabalhar meta de atingir Ponto de Virada|Reconhecer progressos para manter motivação'),

('Destaque', 'Alunos de alto desempenho com engajamento exemplar. Podem ser mentores para outros alunos.',
 'Programa de mentoria para ajudar outros alunos|Desafios acadêmicos adicionais|Preparação para oportunidades de bolsa|Reconhecimento público das conquistas'),

('Avaliar', 'Perfil atípico que não se encaixa em nenhum padrão claro. Requer avaliação individual.',
 'Análise individual do histórico completo|Entrevista com professores que acompanham|Avaliação psicológica se necessário|Definir plano personalizado após análise');
"""


def create_database(db_path: str) -> sqlite3.Connection:
    """Cria banco de dados e tabelas."""
    logger.info(f"[DATABASE] Criando banco de dados: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Cria tabelas
    cursor.executescript(CREATE_ALUNOS_TABLE)
    cursor.executescript(CREATE_INDICES)
    cursor.executescript(CREATE_PERFIS_TABLE)
    cursor.executescript(INSERT_PERFIS)
    
    conn.commit()
    logger.info("[DATABASE] Tabelas criadas com sucesso")
    
    return conn


def load_and_prepare_data(excel_path: str) -> pd.DataFrame:
    """Carrega e prepara dados do Excel."""
    logger.info(f"[DATABASE] Carregando dados de: {excel_path}")
    
    df = pd.read_excel(excel_path)
    
    # Renomeia colunas para snake_case
    column_mapping = {
        'RA': 'ra',
        'Nome': 'nome',
        'Idade 22': 'idade',
        'Gênero': 'genero',
        'Ano nasc': 'ano_nascimento',
        'Ano ingresso': 'ano_ingresso',
        'Instituição de ensino': 'instituicao_ensino',
        'Fase': 'fase',
        'Turma': 'turma',
        'Fase ideal': 'fase_ideal',
        'IAA': 'iaa',
        'IEG': 'ieg',
        'IPS': 'ips',
        'IDA': 'ida',
        'IPV': 'ipv',
        'IAN': 'ian',
        'INDE 22': 'inde',
        'Matem': 'nota_matematica',
        'Portug': 'nota_portugues',
        'Inglês': 'nota_ingles',
        'Defas': 'defasagem',
        'Atingiu PV': 'atingiu_ponto_virada',
        'Pedra 20': 'pedra_2020',
        'Pedra 21': 'pedra_2021',
        'Pedra 22': 'pedra_2022',
        'Indicado': 'indicado_bolsa',
        'Rec Psicologia': 'rec_psicologia',
        'Destaque IEG': 'destaque_ieg',
        'Destaque IDA': 'destaque_ida',
        'Destaque IPV': 'destaque_ipv',
        'grupo': 'cluster_id'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Adiciona perfil baseado no cluster
    if 'cluster_id' in df.columns:
        df['perfil'] = df['cluster_id'].map(CLUSTER_TO_PROFILE)
    else:
        df['cluster_id'] = -1
        df['perfil'] = 'Avaliar'
    
    # Seleciona apenas colunas necessárias
    columns_to_keep = [
        'ra', 'nome', 'idade', 'genero', 'ano_nascimento', 'ano_ingresso',
        'instituicao_ensino', 'fase', 'turma', 'fase_ideal',
        'iaa', 'ieg', 'ips', 'ida', 'ipv', 'ian', 'inde',
        'nota_matematica', 'nota_portugues', 'nota_ingles',
        'defasagem', 'atingiu_ponto_virada',
        'pedra_2020', 'pedra_2021', 'pedra_2022',
        'indicado_bolsa', 'rec_psicologia',
        'destaque_ieg', 'destaque_ida', 'destaque_ipv',
        'cluster_id', 'perfil'
    ]
    
    # Filtra apenas colunas que existem
    columns_to_keep = [c for c in columns_to_keep if c in df.columns]
    df = df[columns_to_keep]
    
    logger.info(f"[DATABASE] Dados carregados: {len(df)} registros, {len(df.columns)} colunas")
    
    return df


def insert_data(conn: sqlite3.Connection, df: pd.DataFrame):
    """Insere dados no banco."""
    logger.info(f"[DATABASE] Inserindo {len(df)} registros...")
    
    df.to_sql('alunos', conn, if_exists='replace', index=False)
    
    conn.commit()
    logger.info("[DATABASE] Dados inseridos com sucesso")


def verify_database(conn: sqlite3.Connection):
    """Verifica se o banco foi criado corretamente."""
    cursor = conn.cursor()
    
    # Conta registros
    cursor.execute("SELECT COUNT(*) FROM alunos")
    total_alunos = cursor.fetchone()[0]
    
    # Conta por perfil
    cursor.execute("SELECT perfil, COUNT(*) FROM alunos GROUP BY perfil ORDER BY COUNT(*) DESC")
    perfis = cursor.fetchall()
    
    logger.info("=" * 50)
    logger.info("[DATABASE] VERIFICAÇÃO DO BANCO")
    logger.info("=" * 50)
    logger.info(f"Total de alunos: {total_alunos}")
    logger.info("Distribuição por perfil:")
    for perfil, count in perfis:
        logger.info(f"  - {perfil}: {count}")
    logger.info("=" * 50)


def setup_database(
    input_path: str = "data/clusterizacao-defasagem-alunos-passos-magicos.xlsx",
    output_path: str = "data/passos_magicos.db"
) -> str:
    """
    Pipeline completo para criar e popular o banco.
    
    Args:
        input_path: Caminho do Excel com dados clusterizados
        output_path: Caminho do banco SQLite de saída
        
    Returns:
        Caminho do banco criado
    """
    # Cria diretório se não existir
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Pipeline
    conn = create_database(output_path)
    df = load_and_prepare_data(input_path)
    insert_data(conn, df)
    verify_database(conn)
    
    conn.close()
    
    logger.info(f"[DATABASE] Banco criado com sucesso: {output_path}")
    
    return output_path



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cria banco SQLite com dados dos alunos")
    parser.add_argument("--input", "-i", default="data/clusterizacao-defasagem-alunos-passos-magicos.xlsx",
                        help="Caminho do Excel de entrada")
    parser.add_argument("--output", "-o", default="data/passos_magicos.db",
                        help="Caminho do banco SQLite de saída")
    
    args = parser.parse_args()
    
    setup_database(args.input, args.output)
