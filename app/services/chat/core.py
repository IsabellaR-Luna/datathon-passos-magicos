# core.py
"""
Classes auxiliares para o serviço de chat com Text-to-SQL.

Classes:
    - SQLGenerator: Gera SQL a partir de texto usando Gemini
    - QueryExecutor: Executa SQL no SQLite
    - ResponseFormatter: Formata resposta amigável
"""

import logging
import sqlite3
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import google.generativeai as genai


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SQLResult:
    """Resultado da geração de SQL."""
    success: bool
    query: Optional[str] = None
    intent: str = "unknown"
    error: Optional[str] = None


@dataclass
class QueryResult:
    """Resultado da execução de query."""
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    error: Optional[str] = None


@dataclass
class ChatResponse:
    """Resposta formatada do chat."""
    message: str
    data: Optional[List[Dict[str, Any]]] = None
    query_used: Optional[str] = None
    intent: str = "unknown"


class ChatError(Exception):
    """Erro base do serviço de chat."""
    pass


class SQLGenerationError(ChatError):
    """Erro na geração de SQL."""
    pass


class QueryExecutionError(ChatError):
    """Erro na execução de query."""
    pass



DATABASE_SCHEMA = """
TABELA: alunos
Descrição: Dados dos alunos da Associação Passos Mágicos

COLUNAS:
- ra: TEXT - Registro do aluno (ex: 'RA-123')
- nome: TEXT - Nome do aluno (ex: 'Aluno-123')
- idade: INTEGER - Idade do aluno
- genero: TEXT - Gênero ('Menino' ou 'Menina')
- ano_nascimento: INTEGER - Ano de nascimento
- ano_ingresso: INTEGER - Ano que entrou no programa
- instituicao_ensino: TEXT - Tipo de escola ('Escola Pública', 'Rede Decisão', etc)
- fase: INTEGER - Fase atual no programa (0-7)
- turma: TEXT - Turma ('A', 'B', 'C', etc)
- fase_ideal: TEXT - Fase ideal para a idade
- iaa: REAL - Indicador de Aprendizagem (0-10)
- ieg: REAL - Indicador de Engajamento (0-10)
- ips: REAL - Indicador Psicossocial (0-10)
- ida: REAL - Indicador de Desempenho Acadêmico (0-10)
- ipv: REAL - Indicador de Ponto de Virada (0-10)
- ian: REAL - Indicador de Adequação de Nível (0-10)
- inde: REAL - Índice de Desenvolvimento Educacional
- nota_matematica: REAL - Nota em matemática (0-10)
- nota_portugues: REAL - Nota em português (0-10)
- nota_ingles: REAL - Nota em inglês (0-10)
- defasagem: INTEGER - Anos de defasagem escolar (negativo = atrasado)
- atingiu_ponto_virada: TEXT - Se atingiu ponto de virada ('Sim' ou 'Não')
- pedra_2020: TEXT - Nível em 2020 ('Quartzo', 'Ágata', 'Ametista', 'Topázio')
- pedra_2021: TEXT - Nível em 2021
- pedra_2022: TEXT - Nível em 2022
- indicado_bolsa: TEXT - Se foi indicado para bolsa ('Sim' ou 'Não')
- rec_psicologia: TEXT - Recomendação psicológica
- destaque_ieg: TEXT - Destaque em engajamento
- destaque_ida: TEXT - Destaque em desempenho
- destaque_ipv: TEXT - Destaque em ponto de virada
- cluster_id: INTEGER - ID do cluster (-1 a 9)
- perfil: TEXT - Perfil do aluno ('Crítico', 'Atenção', 'Em Desenvolvimento', 'Destaque', 'Avaliar')

TABELA: perfis
Descrição: Descrição dos perfis de alunos

COLUNAS:
- nome: TEXT - Nome do perfil
- descricao: TEXT - Descrição do perfil
- recomendacoes: TEXT - Recomendações separadas por '|'

VALORES IMPORTANTES:
- Perfis: 'Crítico', 'Atenção', 'Em Desenvolvimento', 'Destaque', 'Avaliar'
- Pedras (do menor para maior): 'Quartzo', 'Ágata', 'Ametista', 'Topázio'
- Defasagem negativa significa atraso escolar (ex: -2 = 2 anos atrasado)
"""

EXAMPLE_QUERIES = """
EXEMPLOS DE PERGUNTAS E SQL:

1. "Quantos alunos críticos temos?"
   SELECT COUNT(*) as total FROM alunos WHERE perfil = 'Crítico'

2. "Liste os alunos da turma A"
   SELECT nome, perfil, iaa, ieg FROM alunos WHERE turma = 'A'

3. "Qual a média de IAA por perfil?"
   SELECT perfil, ROUND(AVG(iaa), 2) as media_iaa, COUNT(*) as total FROM alunos GROUP BY perfil

4. "Quais alunos têm defasagem maior que 2 anos?"
   SELECT nome, defasagem, perfil FROM alunos WHERE defasagem < -2

5. "Mostre os 5 alunos com maior IEG"
   SELECT nome, ieg, perfil FROM alunos ORDER BY ieg DESC LIMIT 5

6. "Quantos alunos atingiram o ponto de virada?"
   SELECT COUNT(*) as total FROM alunos WHERE atingiu_ponto_virada = 'Sim'

7. "Qual a distribuição de alunos por perfil?"
   SELECT perfil, COUNT(*) as total FROM alunos GROUP BY perfil ORDER BY total DESC

8. "Quais alunos do perfil Atenção têm IEG acima de 8?"
   SELECT nome, ieg, ida FROM alunos WHERE perfil = 'Atenção' AND ieg > 8

9. "Mostre alunos que precisam de acompanhamento psicológico"
   SELECT nome, perfil, rec_psicologia FROM alunos WHERE rec_psicologia = 'Requer avaliação'

10. "Compare médias de notas entre perfis"
    SELECT perfil, ROUND(AVG(nota_matematica), 2) as media_mat, ROUND(AVG(nota_portugues), 2) as media_port FROM alunos GROUP BY perfil
"""


class SQLGenerator:
    """
    Gera SQL a partir de texto usando Google Gemini.
    """
    
    SYSTEM_PROMPT = f"""Você é um assistente especializado em converter perguntas em SQL para um banco de dados de alunos.

    {DATABASE_SCHEMA}

    {EXAMPLE_QUERIES}

    REGRAS:
    1. Retorne APENAS o SQL, sem explicações ou markdown
    2. Use apenas as tabelas e colunas listadas acima
    3. Para filtros de texto, use aspas simples (ex: WHERE perfil = 'Crítico')
    4. Limite resultados grandes com LIMIT 20
    5. Use ROUND() para números decimais
    6. Se a pergunta não for sobre dados, retorne: NAO_SQL

    IMPORTANTE:
    - Perfis são: 'Crítico', 'Atenção', 'Em Desenvolvimento', 'Destaque', 'Avaliar'
    - Defasagem negativa = atraso (ex: -2 significa 2 anos atrasado)
    - Pedras em ordem: Quartzo < Ágata < Ametista < Topázio
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Inicializa o gerador de SQL.
        
        Args:
            api_key: API key do Google
            model_name: Nome do modelo Gemini
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"[SQL GENERATOR] Inicializado com modelo: {model_name}")
    
    def generate(self, question: str) -> SQLResult:
        """
        Gera SQL a partir de uma pergunta.
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            SQLResult com query ou erro
        """
        logger.info(f"[SQL GENERATOR] Gerando SQL para: {question[:50]}...")
        
        try:
            prompt = f"{self.SYSTEM_PROMPT}\n\nPergunta do usuário: {question}\n\nSQL:"
            
            response = self.model.generate_content(prompt)
            sql_text = response.text.strip()
            
            # Remove markdown se presente
            sql_text = self._clean_sql(sql_text)
            
            # Verifica se é uma consulta válida ou NAO_SQL
            if sql_text == "NAO_SQL" or not sql_text:
                logger.info("[SQL GENERATOR] Pergunta não requer SQL")
                return SQLResult(
                    success=True,
                    query=None,
                    intent="conversation"
                )
            
            # Valida SQL básico
            if not self._validate_sql(sql_text):
                logger.warning(f"[SQL GENERATOR] SQL inválido: {sql_text}")
                return SQLResult(
                    success=False,
                    error="SQL gerado é inválido ou potencialmente perigoso"
                )
            
            logger.info(f"[SQL GENERATOR] SQL gerado: {sql_text}")
            
            return SQLResult(
                success=True,
                query=sql_text,
                intent="query"
            )
            
        except Exception as e:
            logger.error(f"[SQL GENERATOR] Erro: {str(e)}")
            return SQLResult(
                success=False,
                error=str(e)
            )
    
    def _clean_sql(self, sql: str) -> str:
        """Remove markdown e limpa o SQL."""
        # Remove blocos de código markdown
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        sql = sql.strip()
        return sql
    
    def _validate_sql(self, sql: str) -> bool:
        """Valida se o SQL é seguro."""
        sql_upper = sql.upper()
        
        # Bloqueia operações perigosas
        dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous:
            if keyword in sql_upper:
                return False
        
        # Deve começar com SELECT
        if not sql_upper.strip().startswith('SELECT'):
            return False
        
        return True



class QueryExecutor:
    """
    Executa queries SQL no banco SQLite.
    """
    
    def __init__(self, db_path: str):
        """
        Inicializa o executor.
        
        Args:
            db_path: Caminho do banco SQLite
        """
        self.db_path = db_path
        logger.info(f"[QUERY EXECUTOR] Inicializado com banco: {db_path}")
    
    def execute(self, query: str) -> QueryResult:
        """
        Executa uma query SQL.
        
        Args:
            query: Query SQL a executar
            
        Returns:
            QueryResult com dados ou erro
        """
        logger.info(f"[QUERY EXECUTOR] Executando: {query[:50]}...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Converte para lista de dicts
            columns = [description[0] for description in cursor.description]
            data = [dict(row) for row in rows]
            
            conn.close()
            
            logger.info(f"[QUERY EXECUTOR] Retornou {len(data)} registros")
            
            return QueryResult(
                success=True,
                data=data,
                columns=columns,
                row_count=len(data)
            )
            
        except Exception as e:
            logger.error(f"[QUERY EXECUTOR] Erro: {str(e)}")
            return QueryResult(
                success=False,
                error=str(e)
            )



class ResponseFormatter:
    """
    Formata respostas amigáveis a partir dos dados.
    """
    
    def __init__(self, api_key: str, model_name: str):
        """
        Inicializa o formatador.
        
        Args:
            api_key: API key do Google
            model_name: Nome do modelo Gemini
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"[RESPONSE FORMATTER] Inicializado com modelo: {model_name}")
    
    def format(
        self,
        question: str,
        query_result: QueryResult,
        sql_used: Optional[str] = None
    ) -> str:
        """
        Formata resposta amigável.
        
        Args:
            question: Pergunta original
            query_result: Resultado da query
            sql_used: SQL utilizado
            
        Returns:
            Resposta formatada em texto
        """
        logger.info("[RESPONSE FORMATTER] Formatando resposta...")
        
        if not query_result.success:
            return f"Desculpe, ocorreu um erro ao buscar os dados: tente novamente mais tarde!"
        
        if query_result.row_count == 0:
            return "Não encontrei nenhum resultado para sua consulta."
        
        # try:
        #     # Para poucos resultados, formata diretamente
        #     if query_result.row_count <= 5:
        #         return self._format_simple(question, query_result)
            
            # Para muitos resultados, usa LLM para resumir
        return self._format_with_llm(question, query_result)
            
        # except Exception as e:
        #     logger.error(f"[RESPONSE FORMATTER] Erro: {str(e)}")
        #     return self._format_simple(question, query_result)
    
    def _format_simple(self, question: str, result: QueryResult) -> str:
        """Formatação simples sem LLM."""
        lines = []
        
        # Se é contagem simples
        if result.row_count == 1 and len(result.columns) <= 2:
            row = result.data[0]
            values = list(row.values())
            if len(values) == 1:
                return f"O resultado é: {values[0]}"
            else:
                parts = [f"{k}: {v}" for k, v in row.items()]
                return f"Resultado: {', '.join(parts)}"
        
        
        lines.append(f"Encontrei {result.row_count} resultado(s):\n")
        
        for i, row in enumerate(result.data[:10], 1):
            parts = [f"{k}: {v}" for k, v in row.items()]
            lines.append(f"{i}. {', '.join(parts)}")
        
        if result.row_count > 10:
            lines.append(f"\n... e mais {result.row_count - 10} resultados.")
        
        return '\n'.join(lines)
    
    def _format_with_llm(self, question: str, result: QueryResult) -> str:
        """Formatação com LLM para resumir."""
        prompt = f"""Você é um assistente pedagógico da Associação Passos Mágicos.

        Você deve considerar as seguintes diretrizes ao elaborar sua resposta:

        1) Leia atentamente a pergunta original do usuario.

        2) Considere todas as informações fornecidas, incluindo nomes corretos e resultados de consultas SQL

        3) Elabore uma resposta clara, concisa e informativa, utilizando linguagem natural baseando-se nas informações levantadas até o momento, nao utilize fontes externas.
        - Use um tom engajante, como se fosse um funcionario novo na empresa que esta querendo mostrar servico.
        - Seja proativo, ou seja, antecipe possiveis interações futuras e pergunte ao usuario se ele gostaria de mais informações ou ajuda adicional.
        - Você nao deve apenas responder, você deve ENTREGAR uma experiéncia completa ao usuario, mantendo-o engajado.
        - Seja criativo na hora de começar a falar, não comece sempre com um "Ótima pergunta", "Olá", etc. Varie a forma como você inicia a conversa para tornar a experiência mais agradável e menos robotica.

        4) Se a pergunta do usuario nao puder ser respondida com as informações disponiveis, informe educadamente que nao foi possivel encontrar a resposta.

        5) NUNCA mencione os nomes dos agentes, códigos ou detalhes técnicos sobre o funcionamento do sistema na sua resposta ao usuario.

        6) Revise sua resposta para garantir precisao e clareza antes de envia-la ao usuario.

        7) Fique atento as saudações e formas de se comunicar (Olá, Oi!, etc), lembre-se que voce é um assistente pedagógico da Associação Passos Mágicos, e deve manter um tom amigável, engajante e proativo.

        8) Se a pergunta do usuário for apenas uma saudação (Ex: "Olá", "Oi", "Bom dia"), responda de forma amigável e engajante, sem mencionar que é um assistente ou detalhes técnicos. Exemplo de resposta: "Olá! Como posso ajudar você hoje? Se tiver alguma dúvida sobre os alunos ou precisar de informações, é só perguntar!"

        O professor perguntou: "{question}"

        Os dados retornados foram:
        {result.data}

        Total de registros: {result.row_count}
        Colunas: {result.columns}


        Observações:
        NAO responda perguntas fora do escopo. NAO dê ao usuário informações que nao estejam relacionadas a esse tema.

        """
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def format_conversation(self, question: str) -> str:
        """
        Responde perguntas que não são sobre dados.
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Resposta conversacional
        """
        logger.info("[RESPONSE FORMATTER] Gerando resposta conversacional...")
        
        prompt = f"""Você é um assistente pedagógico da Associação Passos Mágicos.
        Você ajuda professores a entender e apoiar alunos em situação de vulnerabilidade social.

        O professor disse: "{question}"

        Responda de forma útil e amigável. Se a pergunta for sobre dados de alunos,
        sugira que ele faça uma pergunta mais específica como:
        - "Quantos alunos temos no perfil Crítico?"
        - "Quais alunos da turma A precisam de atenção?"
        - "Qual a média de engajamento por perfil?"
        """
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def format_error(self, error: str) -> str:
        """Formata mensagem de erro amigável."""
        return f"""Desculpe, não consegui processar sua solicitação.

        Tente reformular sua pergunta. Exemplos:
        - "Quantos alunos estão no perfil Crítico?"
        - "Liste os alunos da turma B"
        - "Qual a média de IAA dos alunos Destaque?"
        """