from semantic_kernel.contents import AuthorRole, ChatMessageContent
from semantic_kernel.agents import Agent, AgentResponseItem
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatHistoryAgentThread

from texttosql.infra.tools.sqlitetool import DataBaseManager
import re
import json

QUERY_EXECUTOR_DESCRIPTION = """
Executa a query SQL gerada pelo agente Query Generator na base de dados e retorna os resultados
"""

class QueryExecutorAgent(Agent):

    def __init__(self, name: str):
        super().__init__(name=name, description=QUERY_EXECUTOR_DESCRIPTION)

    async def get_response(self, chat_history, **kwargs):
        result = self.handler_func(chat_history)
        return ChatMessageContent(
            role=AuthorRole.ASSISTANT,
            name=self.name,
            content=result
        )

    async def invoke(self, chat_history, thread, **kwargs):
        thread = await self._ensure_thread_exists_with_messages(
            messages=chat_history,
            thread=thread,
            construct_thread=lambda: ChatHistoryAgentThread(),
            expected_type=ChatHistoryAgentThread,
        )
        assert thread.id is not None  # nosec
        response = await self.get_response(chat_history, **kwargs)
        yield AgentResponseItem(message=response, thread=thread)

    async def invoke_stream(self, chat_history, thread, **kwargs):
        thread = await self._ensure_thread_exists_with_messages(
            messages=chat_history,
            thread=thread,
            construct_thread=lambda: ChatHistoryAgentThread(),
            expected_type=ChatHistoryAgentThread,
        )
        assert thread.id is not None  # nosec
        response = await self.get_response(chat_history, **kwargs)
        yield AgentResponseItem(message=response, thread=thread)

    async def on_message_impl(self, message, ctx):
        chat_history = ctx.get("chat_history", None)
        result_text = self.handler_func(chat_history)
        return ChatMessageContent(
            role=AuthorRole.ASSISTANT,
            name=self.name,
            content=result_text
        )

    def handler_func(self, chat_history):
        # Implement the logic to handle the message and generate a response
        # Roda a query SQL da ultima mensagem do historico, se houver
        print("ULTIMA MENSAGEM:", chat_history[-1])
        last_message = chat_history[-1].content
        # if last_message.name == "Executor" and last_message.role == AuthorRole.ASSISTANT:
        print("Executando a query SQL da Ultima mensagem do Executor.")
        print(f"Resposta do Executor: {last_message}")
        # Arrumar a query SQL que esta dentro do JSON
        last_message = self.extract_query_from_text(last_message)
        print(f"Query SQL: {last_message}")
        try:
            query_result = DataBaseManager.execute_query(last_message)
            print(f"Resultado da query SQL:\n{query_result}")
            print("#######################")
            return f"Resultado da query SQL:\n{query_result}"
        except Exception as e:
            error_message = f"Erro ao executar a query SQL: {e}"
            print(error_message)
            return error_message
        # else:
        #     print("A Ultima mensagem nao é do Executor ou nao é uma mensagem de assistente.")
        #     return "Erro: A Ultima mensagem nao é do Executor ou nao é uma mensagem de assistente."

    def extract_query_from_text(self, text: str) -> str:
        extracted_query = None
        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_content = match.group(0)
            try:
                data = json.loads(json_content)
                if "sql" in data:
                    extracted_query = data["sql"]
            except json.JSONDecodeError as e:
                pass
        # Se não conseguiu extrair via JSON, tenta extrair via regex simples
        # Query começa no SELECT e vai até o ;
        if extracted_query is None:
            match = re.search(r'(SELECT .*?;)', text, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_query = match.group(1).strip()
            else:
                extracted_query = ""
                
        return extracted_query