from typing import Annotated
from chromadb.api.types import EmbeddingFunction
import logging
import csv
import io
from shared.config.settings import settings
from openai import AzureOpenAI
import chromadb
import sqlite3
import os


TABLE_NAME = f"{settings.DB_SCHEMA_NAME}"

DB_SCHEMA = f"""
O Banco de Dados é SQLite, atente-se para utilizar a sintaxe correta. (arquivo: {settings.DB_SCHEMA_NAME})

Tabela disponível:
    Nome: resultados
    Descrição: Essa tabela contém os resultados dos indicadores ESG para as empresas na base.
    Colunas:
        - empresa: TEXT // Nome da empresa
        - nome_pregao: TEXT // Nome de negociação da empresa na bolsa
        - categoria: TEXT // Categoria do indicador ESG
        - indicador: TEXT // Nome do indicador ESG
        - medida: TEXT // Unidade de medida do indicador
        - descricao: TEXT // Descrição detalhada do indicador ESG
        - resultado: TEXT // Valor do indicador ESG para a empresa
        - pagina: TEXT // Página do documento de origem do dado
        - ano_referencia: BIGINT // Ano de referência do resultado
        - nome_plataforma: TEXT // Nome cadastrado na plataforma ESG
        - cnpj: TEXT // CNPJ da empresa
        - setor: TEXT // Setor econômico da empresa
        - subsetor: TEXT // Subsetor econômico da empresa
        - segmento: TEXT // Segmento econômico da empresa
        - setor_ise: TEXT // Setor conforme classificação ISE B3
        - razao_social: TEXT // Nome jurídico completo da empresa
        - denominacao: TEXT // Denominação social da empresa
"""





class DataBaseManager:
    logger: logging.Logger

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @classmethod
    def configure(cls, logger: logging.Logger):
        cls.logger = logger

    @staticmethod
    def connect_to_db():
        """Conecta ao banco SQLite local."""
        
        try:
            db_path = settings.DB_SCHEMA_NAME
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            return conn, cursor

        
        except Exception as e:
            logging.getLogger("ChatbotAPI").error(f"Erro ao conectar ao banco SQLite: {e}")
            print(f"Erro ao conectar ao banco SQLite: {e}")
            return None, None


    @staticmethod
    def execute_query(query: str) -> Annotated[str, "The database query result."]:
        logging.getLogger("ChatbotAPI").info(f"[DB TOOL QUERY] Conectando ao banco SQLite local")
        logging.getLogger("ChatbotAPI").info(f"[DB TOOL QUERY] Query a ser executada: {query}")
        
        print(f"[DB TOOL QUERY] Conectando ao banco SQLite local")
        print(f"[DB TOOL QUERY] Query a ser executada: {query}")
        
        conn, cursor = DataBaseManager.connect_to_db()
        
        if not conn or not cursor:
            return None
        
        try:
            cursor.execute(query)
            
            results = cursor.fetchall()
            
            # Obtém o nome das colunas
            column_names = [desc[0] for desc in cursor.description]
            
            # Cria um objeto StringIO para escrever o CSV em uma string
            output = io.StringIO()
            csv_writer = csv.writer(output)
            
            # Escreve os nomes das colunas no csv
            csv_writer.writerow(column_names)
            
            # Escreve os resultados no csv
            csv_writer.writerows(results)
            
            # Obtém o conteúdo como uma string
            csv_output = output.getvalue()
            
            # Fecha o objeto stringIO
            output.close()
            cursor.close()
            conn.close()
            
            return csv_output
        
        except Exception as e:
            logging.getLogger("ChatbotAPI").error(f"Erro ao executar query no SQLite: {e}")
            print(f"Erro ao executar query no SQLite: {e}")
            
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            return None


    @staticmethod
    def indicador_lookup(name: str) -> Annotated[str, "A list of up to 5 candidate ESG indicators matching the provided name."]:
        """Busca semântica de indicadores usando ChromaDB e SQLite, mantendo retorno CSV."""
        
        TOPK = 5
        logger = logging.getLogger("ChatbotAPI")
        logger.info(f"[DB TOOL INDICADOR LOOKUP] Usando ChromaDB e SQLite. Indicador buscado: {name}")
        print(f"[DB TOOL INDICADOR LOOKUP] Usando ChromaDB e SQLite. Indicador buscado: {name}")


        try:
            chroma_client = chromadb.PersistentClient(path="chromadb_data")
            
            collection = chroma_client.get_collection(
                name="indicadores_collection",
                embedding_function=CustomEmbeddingFunction()
            )
            
            # Transforma o texto do usuário em vetor
            query_vec = get_embedding(name)
            
            if not query_vec:
                raise ValueError("Embedding da query retornou vazio")
            
            
            # Aqui é basicamente : "Aqui está o significado vetorial da pergunta do usuário. Me traga os indicadores semanticamente mais próximos."
            res = collection.query(
                query_embeddings=[query_vec],
                n_results=TOPK
            )
        except Exception as e:
            logger.error(f"Erro ao consultar ChromaDB: {e}")
            return None
        

        # Monta o resultado no formato CSV 
        output = io.StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerow(["indicador", "medida", "palavras_chaves"])
          
        for doc, meta, dist in zip(
            res["documents"][0], 
            res["metadatas"][0], 
            res["distances"][0]
        ):
            
            csv_writer.writerow([
                meta.get("indicador", ""),
                meta.get("medida", ""),
                meta.get("palavras_chaves", ""),
            ])
            
        csv_output = output.getvalue()
        output.close()
        logger.info(f"[DB TOOL INDICADOR LOOKUP] Resultado da busca: {csv_output}")
        print(f"[DB TOOL INDICADOR LOOKUP] Resultado da busca: {csv_output}")
        
        return csv_output



    @staticmethod
    def empresas_lookup(name: str) -> Annotated[str, "A list of up to 10 candidate companies matching the provided name."]:
        logging.getLogger("ChatbotAPI").info(f"[DB TOOL EMPRESAS LOOKUP] Usando SQlite.")
        logging.getLogger("ChatbotAPI").info(f"[DB TOOL EMPRESAS LOOKUP] Empresa buscada: {name}")
        
        print(f"[DB TOOL EMPRESAS LOOKUP] Usando SQlite.")
        print(f"[DB TOOL EMPRESAS LOOKUP] Empresa buscada: {name}")
        
        
        # E se eu colocar um outro OR LOWER, e por nome_plataforma ... Avaliar amanhã
        query = f"""
        SELECT DISTINCT empresa, nome_plataforma, cnpj, setor, subsetor,
        segmento, setor_ise, razao_social, nome_pregao, denominacao
        FROM resultados
        WHERE LOWER(empresa) LIKE LOWER('%{name}%')
        OR LOWER(razao_social) LIKE LOWER('%{name}%')
        OR LOWER(denominacao) LIKE LOWER('%{name}%')
        OR LOWER(nome_pregao) LIKE LOWER('%{name}%')
        OR LOWER(nome_plataforma) LIKE LOWER('%{name}%')
        LIMIT 10;
        """
        return DataBaseManager.execute_query(query)




class CustomEmbeddingFunction(EmbeddingFunction):
    """ Classe customizada para chamada do método get_embeddings, adaptada para ChromaDB. """

    def __call__(self, input):
        return [get_embedding(text) for text in input]

        
    
client = AzureOpenAI(
    api_key=settings.APIKEY,
    azure_endpoint=settings.ENDPOINT,
    api_version="2024-12-01-preview",
)

def get_embedding(text):
    
    response = client.embeddings.create(
        input=text,
        model="embedding"
    )

    embedding = response.data[0].embedding

    if not isinstance(embedding, list):
        raise ValueError("Embedding inválido retornado pelo Azure")

    return embedding
    