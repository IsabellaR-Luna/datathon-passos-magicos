# chatbot_ui.py
"""
Interface Streamlit para o chatbot.

Uso:
    streamlit run chatbot_ui.py
"""

import streamlit as st
import requests


API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Passos Mágicos - Chatbot",
    page_icon="🎓",
    layout="centered"
)


def send_message(pergunta: str) -> dict:
    """Envia pergunta para a API."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"pergunta": pergunta},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"resposta": "❌ Erro: API não está rodando. Execute: uvicorn app.main:app --reload", "tipo": "error"}
    except Exception as e:
        return {"resposta": f"❌ Erro: {str(e)}", "tipo": "error"}


def get_suggestions() -> list:
    """Busca sugestões de perguntas."""
    try:
        response = requests.get(f"{API_URL}/chat/suggestions", timeout=5)
        response.raise_for_status()
        return response.json().get("sugestoes", [])
    except:
        return [
            "Quantos alunos temos em cada perfil?",
            "Quais alunos estão no perfil Crítico?",
            "Qual a média de IAA por perfil?"
        ]


# Header
st.title("🎓 Passos Mágicos")
st.subheader("Assistente Pedagógico")
st.markdown("---")

# Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Mostra SQL se houver
        if message.get("sql"):
            with st.expander("🔍 SQL utilizado"):
                st.code(message["sql"], language="sql")

# Sugestões (só mostra se não tem histórico)
if not st.session_state.messages:
    st.markdown("**💡 Experimente perguntar:**")
    
    suggestions = get_suggestions()
    cols = st.columns(2)
    
    for i, suggestion in enumerate(suggestions[:6]):
        col = cols[i % 2]
        if col.button(suggestion, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending_question = suggestion
            st.rerun()

# Processa pergunta pendente (das sugestões)
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
    
    # Adiciona pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Envia para API
    with st.spinner("Pensando..."):
        response = send_message(question)
    
    # Adiciona resposta ao histórico
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.get("resposta", "Sem resposta"),
        "sql": response.get("sql_utilizado")
    })
    
    st.rerun()

# Input do usuário
if prompt := st.chat_input("Digite sua pergunta..."):
    # Adiciona pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Envia para API
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = send_message(prompt)
        
        st.markdown(response.get("resposta", "Sem resposta"))
        
        # Mostra SQL se houver
        if response.get("sql_utilizado"):
            with st.expander("🔍 SQL utilizado"):
                st.code(response["sql_utilizado"], language="sql")
    
    # Adiciona resposta ao histórico
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.get("resposta", "Sem resposta"),
        "sql": response.get("sql_utilizado")
    })

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ Sobre")
    st.markdown("""
    Este chatbot ajuda professores a consultar 
    dados dos alunos da Associação Passos Mágicos.
    
    **Exemplos de perguntas:**
    - Quantos alunos críticos temos?
    - Liste alunos da turma A
    - Qual a média de IEG por perfil?
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Perfis de alunos:**")
    st.markdown("""
    - 🔴 **Crítico**: Risco alto
    - 🟠 **Atenção**: Estagnados
    - 🟡 **Em Desenvolvimento**: No caminho
    - 🟢 **Destaque**: Alto desempenho
    """)