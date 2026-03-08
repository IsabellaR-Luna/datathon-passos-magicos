# datathon-passos-magicos
Sistema de análise de alunos com clustering e chatbot para a Associação Passos Mágicos.

## 📋 Sobre o Projeto

Este projeto foi desenvolvido para o **Datathon PÓS TECH** com o objetivo de apoiar a Associação Passos Mágicos na identificação e acompanhamento de alunos em situação de vulnerabilidade social.

### Funcionalidades

- **🔍 Clustering de Alunos**: Identificação automática de perfis (Crítico, Atenção, Em Desenvolvimento, Destaque)
- **💬 Chatbot com Text-to-SQL**: Interface conversacional para consultas ao banco de dados
- **📊 API REST**: Endpoints para integração com outros sistemas
- **📈 Monitoramento de Drift**: Detecção de mudanças na distribuição dos dados

## 📁 Estrutura do Projeto

```datathon-passos-magicos/         
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── cluster.py
│   │   └── chat.py
│   └── services/
│       ├── __init__.py
│       ├── chat_service.py
│       ├── cluster_service.py
│       ├── chat/
│       │   ├── __init__.py
│       │   └── core.py
│       └── clustering/
│           ├── __init__.py
│           └── core.py
├── tests/                        
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_cluster.py
│   ├── test_chat.py
│   ├── test_api.py
│   ├── test_cluster_service.py
│   └── test_chat_service.py
├── data/
│   └── passos_magicos.db
├── scripts/
│   └── database_setup.py

```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11+
- API Key do Google (Gemini)

### Instalação Local

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/passos-magicos.git
cd passos-magicos

# 2. Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Configure variáveis de ambiente
cp .env.example .env
# Edite .env e adicione sua GOOGLE_API_KEY

# 5. Inicie a API
uvicorn app.main:app --reload --port 8000

# 6. Em outro terminal, inicie o Streamlit
streamlit run chatbot_ui.py
```

### Com Docker

```bash
# 1. Configure variáveis de ambiente
cp .env.example .env
# Edite .env e adicione sua GOOGLE_API_KEY

# 2. Suba os containers
docker-compose up --build

# 3. Acesse
# API: http://localhost:8000
# UI:  http://localhost:8501
```

## 📡 API Endpoints

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/health` | GET | Health check |
| `/clusters/summary` | GET | Estatísticas dos perfis |
| `/clusters/students` | GET | Lista alunos (com filtros) |
| `/clusters/student/{ra}` | GET | Dados de um aluno |
| `/clusters/profiles` | GET | Lista perfis disponíveis |
| `/chat` | POST | Pergunta ao chatbot |
| `/chat/suggestions` | GET | Sugestões de perguntas |

### Exemplos

```bash
# Health check
curl http://localhost:8000/health

# Resumo dos clusters
curl http://localhost:8000/clusters/summary

# Alunos do perfil Crítico
curl "http://localhost:8000/clusters/students?perfil=Crítico"

# Pergunta ao chatbot
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pergunta": "Quantos alunos temos em cada perfil?"}'
```

## 💬 Exemplos de Perguntas para o Chatbot

- "Quantos alunos temos em cada perfil?"
- "Quais alunos do perfil Crítico precisam de atenção?"
- "Qual a média de engajamento (IEG) por turma?"
- "Liste os alunos que atingiram o Ponto de Virada"
- "Quais alunos têm maior defasagem escolar?"

## 🧪 Testes

```bash
# Rodar todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=app --cov-report=term-missing

# Gerar relatório HTML
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

## 🚢 Deploy no Render

1. Faça fork deste repositório
2. Conecte ao Render
3. Configure a variável `GOOGLE_API_KEY` no painel do Render
4. Deploy automático via `render.yaml`

## 📊 Perfis de Alunos

| Perfil | Descrição | Recomendações |
|--------|-----------|---------------|
| 🔴 **Crítico** | Aprendizado muito baixo | Intervenção urgente, avaliação psicopedagógica |
| 🟠 **Atenção** | Estagnados há muito tempo | Tutoria em pequenos grupos, monitoramento |
| 🟡 **Em Desenvolvimento** | No caminho certo | Manter acompanhamento, incentivar |
| 🟢 **Destaque** | Alto desempenho | Programa de mentoria, desafios extras |
| ⚪ **Avaliar** | Perfil atípico | Análise individual necessária |

## 🛠️ Tecnologias

- **Backend**: FastAPI, Pydantic
- **ML**: scikit-learn, UMAP, HDBSCAN
- **LLM**: Google Gemini (Text-to-SQL)
- **Database**: SQLite
- **Frontend**: Streamlit
- **Deploy**: Docker, Render

