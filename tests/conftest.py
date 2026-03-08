# conftest.py
"""
Fixtures compartilhadas para os testes.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import sqlite3
from pathlib import Path


@pytest.fixture
def sample_df():
    """DataFrame de exemplo com dados de alunos."""
    return pd.DataFrame({
        'RA': ['RA-1', 'RA-2', 'RA-3', 'RA-4', 'RA-5'],
        'Nome': ['Aluno-1', 'Aluno-2', 'Aluno-3', 'Aluno-4', 'Aluno-5'],
        'IAA': [8.3, 0.0, 7.5, 9.0, 4.2],
        'IEG': [4.1, 7.9, 8.0, 9.4, 5.5],
        'IPS': [5.6, 5.6, 6.5, 7.5, 6.0],
        'IDA': [4.0, 5.6, 6.8, 9.3, 3.5],
        'IPV': [7.2, 7.5, 6.7, 7.2, 5.8],
        'IAN': [5.0, 10.0, 10.0, 5.0, 5.0],
        'Defas': [-1, 0, 0, -2, -1]
    })


@pytest.fixture
def sample_df_large():
    """DataFrame maior para testes de clustering."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'RA': [f'RA-{i}' for i in range(n_samples)],
        'Nome': [f'Aluno-{i}' for i in range(n_samples)],
        'IAA': np.random.uniform(0, 10, n_samples),
        'IEG': np.random.uniform(0, 10, n_samples),
        'IPS': np.random.uniform(2.5, 10, n_samples),
        'IDA': np.random.uniform(0, 10, n_samples),
        'IPV': np.random.uniform(0, 10, n_samples),
        'IAN': np.random.uniform(0, 10, n_samples),
        'Defas': np.random.randint(-5, 3, n_samples)
    })


@pytest.fixture
def sample_df_with_nulls():
    """DataFrame com valores nulos."""
    return pd.DataFrame({
        'IAA': [8.3, None, 7.5, 9.0, None],
        'IEG': [4.1, 7.9, None, 9.4, 5.5],
        'IPS': [5.6, 5.6, 6.5, None, 6.0],
        'IDA': [4.0, 5.6, 6.8, 9.3, 3.5],
        'IPV': [7.2, 7.5, 6.7, 7.2, 5.8],
        'IAN': [5.0, 10.0, 10.0, 5.0, 5.0],
        'Defas': [-1, 0, 0, -2, -1]
    })


@pytest.fixture
def sample_features():
    """Dicionário com features de um aluno."""
    return {
        'IAA': 7.5,
        'IEG': 8.0,
        'IPS': 6.5,
        'IDA': 6.0,
        'IPV': 7.0,
        'IAN': 10.0,
        'Defas': 0
    }


@pytest.fixture
def temp_db():
    """Banco de dados SQLite temporário."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Cria tabelas
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE alunos (
            ra TEXT PRIMARY KEY,
            nome TEXT,
            idade INTEGER,
            turma TEXT,
            iaa REAL,
            ieg REAL,
            ips REAL,
            ida REAL,
            ipv REAL,
            ian REAL,
            defasagem INTEGER,
            cluster_id INTEGER,
            perfil TEXT,
            atingiu_ponto_virada TEXT
        )
    """)
    
    # Insere dados de teste
    alunos = [
        ('RA-1', 'Aluno-1', 15, 'A', 8.5, 9.0, 7.5, 8.0, 7.5, 10.0, 0, 5, 'Destaque', 'Sim'),
        ('RA-2', 'Aluno-2', 14, 'A', 0.0, 4.0, 5.0, 3.0, 5.0, 5.0, -2, 2, 'Crítico', 'Não'),
        ('RA-3', 'Aluno-3', 16, 'B', 7.0, 7.5, 6.5, 6.0, 6.5, 10.0, 0, 1, 'Em Desenvolvimento', 'Não'),
        ('RA-4', 'Aluno-4', 15, 'B', 6.0, 6.0, 6.0, 5.0, 5.5, 5.0, -1, 3, 'Atenção', 'Não'),
        ('RA-5', 'Aluno-5', 13, 'A', 9.0, 9.5, 8.0, 9.0, 8.5, 10.0, 0, 7, 'Destaque', 'Sim'),
    ]
    
    cursor.executemany(
        "INSERT INTO alunos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        alunos
    )
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_api_key():
    """API key fake para testes."""
    return "fake-api-key-for-testing"
