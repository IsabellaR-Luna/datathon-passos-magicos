import json
import logging
import re
import sys
import time
import os
from typing import Any, List, Optional, AsyncGenerator, Annotated
from datetime import datetime
from shared.config.settings import settings
from texttosql.domain.entities.query_output import QueryOutput
from texttosql.domain.interfaces.ai_agent_interface import AIAgentInterface
from texttosql.infra.ai.agents.gerente_agente import TOPIC, SELETOR_INSTRUCTIONS
from texttosql.infra.ai.agents.empresa_agente import EMPRESA_DESCRIPTION, EMPRESA_INSTRUCTIONS, EmpresaPlugins
from texttosql.infra.ai.agents.indicador_agente import BuscaIndicadorAgent
from texttosql.infra.ai.agents.busca_empresa_agente import BuscaEmpresaAgent
from texttosql.infra.ai.agents.chart_planner_agente import CHART_PLANNER_DESCRIPTION, CHART_PLANNER_INSTRUCTION
from texttosql.infra.ai.agents.chart_renderer_agente import ChartRendererAgent
from texttosql.infra.ai.agents.desambiguador_agente import DESAMBIGUADOR_DESCRIPTION, DESAMBIGUADOR_INSTRUCTIONS
from texttosql.infra.ai.agents.query_executor_agente import QueryExecutorAgent, QUERY_EXECUTOR_DESCRIPTION
from texttosql.infra.ai.agents.query_generator_agente import QUERY_GENERATOR_DESCRIPTION, QUERY_GENERATOR_INSTRUCTIONS
from texttosql.infra.ai.agents.enriquecedor_agente import ENRIQUECEDOR_DESCRIPTION, ENRIQUECEDOR_INSTRUCTION
from texttosql.infra.ai.agents.dialogue_agente import DIALOGUE_DESCRIPTION, DIALOGUE_INSTRUCTIONS
from texttosql.infra.tools.sqlitetool import DataBaseManager, DB_SCHEMA
from texttosql.infra.ai.resilient_chatcompletion import ResilientChatCompletionService

from semantic_kernel.agents import Agent, ChatCompletionAgent, GroupChatOrchestration
from semantic_kernel.agents.orchestration.group_chat import BooleanResult, GroupChatManager, MessageResult, StringResult
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig

from texttosql.infra.ai.azure_openai_client import AzureOpenAIChatCompletionClient
from texttosql.infra.observability.presentation_console import flow_event, agent_step, agent_output
import traceback

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

_raw_azure_service = AzureOpenAIChatCompletionClient(
    service_id=settings.SERVICEID,
    deployment_name=settings.GPTMODEL,
    endpoint=settings.ENDPOINT,
    api_key=settings.APIKEY,
    api_version=settings.APIVERSION,
)

shared_chat_completion_service = ResilientChatCompletionService(_raw_azure_service)
flow_logger = logging.getLogger("ChatbotAPI")

def get_agents() -> list[Agent]:
    """Return a list of agents that will participate in the group style discussion.
    Feel free to add or remove agents.
    """
    default_settings = AzureChatPromptExecutionSettings(
        temperature=0.0,
        max_tokens=4096
    )

    identificaempresas = ChatCompletionAgent(
        name="IdentificaEmpresas",
        description=EMPRESA_DESCRIPTION,
        instructions=EMPRESA_INSTRUCTIONS,
        service=shared_chat_completion_service,
        arguments=KernelArguments(default_settings),
        plugins=[EmpresaPlugins()]
    )

    buscaempresas = BuscaEmpresaAgent(
        name="BuscaEmpresas",
        arguments=KernelArguments(default_settings)
    )

    buscaindicadores = BuscaIndicadorAgent(
        name="BuscaIndicadores",
        arguments=KernelArguments(default_settings)
    )

    chart_planner = ChatCompletionAgent(
        name="ChartPlanner",
        description=CHART_PLANNER_DESCRIPTION,
        instructions=CHART_PLANNER_INSTRUCTION,
        service=shared_chat_completion_service,
        arguments=KernelArguments(default_settings),
    )

    chart_renderer = ChartRendererAgent(
        name="ChartRenderer"
    )

    desambiguador = ChatCompletionAgent(
        name="Desambiguador",
        description=DESAMBIGUADOR_DESCRIPTION,
        instructions=DESAMBIGUADOR_INSTRUCTIONS,
        service=shared_chat_completion_service,
        arguments=KernelArguments(default_settings),
    )

    query_generator = ChatCompletionAgent(
        name="QueryGenerator",
        description=QUERY_GENERATOR_DESCRIPTION,
        instructions=QUERY_GENERATOR_INSTRUCTIONS,
        service=shared_chat_completion_service,
        arguments=KernelArguments(default_settings),
    )

    executor = QueryExecutorAgent(name="Executor")

    enriquecedor = ChatCompletionAgent(
        name="Enriquecedor",
        description=ENRIQUECEDOR_DESCRIPTION,
        instructions=ENRIQUECEDOR_INSTRUCTION,
        service=shared_chat_completion_service,
        arguments=KernelArguments(default_settings),
    )

    dialogue = ChatCompletionAgent(
        name="Dialogue",
        description=DIALOGUE_DESCRIPTION,
        instructions=DIALOGUE_INSTRUCTIONS,
        service=shared_chat_completion_service,
        arguments=KernelArguments(default_settings),
    )

    return [
        identificaempresas, buscaempresas, buscaindicadores,
        desambiguador, query_generator, chart_planner, chart_renderer,
        executor, enriquecedor, dialogue
    ]

class ChatCompletionGroupChatManager(GroupChatManager):
    """Simple chat completion base group chat manager.
    This chat completion service requires a model that supports structured output.
    """

    service: ChatCompletionClientBase
    topic: str
    selection_prompt: str = SELETOR_INSTRUCTIONS
    process_queue: Any = None
    query_id: Any = None

    def __init__(self, topic: str, service: ChatCompletionClientBase, **kwargs) -> None:
        """Initialize the group chat manager."""
        super().__init__(topic=topic, service=service, **kwargs)

    async def _render_prompt(self, prompt: str, arguments: KernelArguments) -> str:
        """Helper to render a prompt with arguments"""
        prompt_template_config = PromptTemplateConfig(template=prompt)
        prompt_template = KernelPromptTemplate(prompt_template_config=prompt_template_config)
        return await prompt_template.render(Kernel(), arguments=arguments)

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """Provide concrete implementation for determining if user input is needed.
        The manager will check if input from human is needed after each agent message.
        """
        return BooleanResult(result=False, reason="No user input needed.")

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """Provide concrete implementation for determining if the discussion should end.
        The manager will check if the conversation should be terminated after each agent message
        or human input (if applicable).
        """
        should_terminate = await super().should_terminate(chat_history)
        if should_terminate.result:
            return should_terminate

        last_message = chat_history.messages[-1]
        
        if last_message.name == "Dialogue":
            return BooleanResult(result=True, reason="A pergunta foi respondida.")
        return BooleanResult(result=False, reason="A pergunta ainda nao foi respondida.")
    

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringResult:
        """Provide concrete implementation for selecting the next agent to speak.
        The manager will select the next agent to speak after each agent message
        or human input (if applicable) if the conversation is not terminated.
        """
        chat_history.messages.insert(
            0,
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.selection_prompt,
                    KernelArguments(
                        topic=self.topic,
                        participants="\n".join([f"{k}: {v}" for k, v in participant_descriptions.items()]),
                    )
                )
            )
        )

        # Adiciona a pergunta do usuario ao historico, caso ele esteja vazio
        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content="Selecione o proximo participante a falar.")
        )

        local_logger = logging.getLogger("multiagent.select_next_agent")

        try:
            response = await self.service.get_chat_message_content(
                chat_history,
                settings=AzureChatPromptExecutionSettings(
                    response_format=StringResult,
                    temperature=0.0,
                    max_tokens=1024
                ),
            )
        except Exception as e:
            local_logger.error(f"Failed to get chat message content: {e}")
            raise

        raw_content = response.content.strip()

        if raw_content.startswith("```json"):
            raw_content = raw_content.removeprefix("```json").removesuffix("```").strip()
        if not raw_content.startswith("{"):
            response = json.dumps({
                "result": raw_content,
                "reason": ""
            })
            participant_name_with_reason = StringResult.model_validate_json(response)
        else:
            # transforma raw_content em json
            response = json.dumps(json.loads(raw_content))
            participant_name_with_reason = StringResult.model_validate_json(response)

        agent_step(
            flow_logger,
            agent_name=participant_name_with_reason.result,
            reason=participant_name_with_reason.reason,
        )

        if os.getenv("CHATBOT_DEBUG_QUERYGEN", "0") == "1" and participant_name_with_reason.result == "QueryGenerator":
            flow_event(
                flow_logger,
                stage="debug",
                message=f"QueryGenerator selecionado com {len(chat_history.messages)} mensagens no histórico.",
            )
        
        if self.process_queue:
            try:
                self.process_queue.update(self.query_id, participant_name_with_reason.result, "")
            except Exception as e:
                flow_event(flow_logger, stage="queue", message=f"Falha ao atualizar fila: {e}", status="warning", query_id=self.query_id)

        if participant_name_with_reason.result in participant_descriptions:
            return participant_name_with_reason

        raise RuntimeError(f"Unknown participant selected: {response.content}.")
    

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> MessageResult:
        """Provide concrete implementation for filtering the results of the discussion.
        The manager will filter the results of the conversation after the conversation is terminated.
        """
        if not chat_history.messages:
            raise RuntimeError("No messages in the chat history.")

        last_message = chat_history.messages[-1]

        if last_message.name == "Dialogue":
            return MessageResult(
                result=ChatMessageContent(role=AuthorRole.ASSISTANT, content=last_message.content),
                reason="A pergunta foi respondida."
            )
        return MessageResult(
            result=ChatMessageContent(role=AuthorRole.ASSISTANT, content="Desculpe, ocorreu algum problema e nao consegui achar a resposta."),
            reason="A pergunta nao foi respondida."
        )

def agent_response_callback(message: ChatMessageContent) -> None:
    """Callback function to retrieve agent responses."""
    agent_output(flow_logger, agent_name=message.name or "unknown", content=message.content or "")

class SKAgentImplementation(AIAgentInterface):
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._max_try = 2
        self._time_out = 5
        self.data = datetime.now().strftime("%Y%m%d_%H%M%S")

        DataBaseManager.configure(logger)
        self.table_schema = DB_SCHEMA

    def prompt_wrapper(self, query: str, summary: str) -> str:
        prompt = f"""
        Resumo da conversa até agora:
        `{summary}`

        Pergunta do usuario:
        #####
        `{query}`
        #####

        JSON do assistente:
        """
        return prompt

    async def _message(self, query: str, summary: str, query_id, process_queue) -> QueryOutput:
        """
        Envia uma mensagem ao agente e retorna a resposta.

        Args:
            query: Mensagem do usuario.

        Returns:
            QueryOutput: Resposta do agente.
        """
        query = self.prompt_wrapper(query, summary)

        flow_event(self.logger, stage="message", message="Iniciando orquestração principal.", query_id=query_id)

        self.agents = get_agents()
        self.group_chat_orchestration = GroupChatOrchestration(
            members=self.agents,
            manager=ChatCompletionGroupChatManager(
                topic=TOPIC,
                service=shared_chat_completion_service,
                max_rounds=20,
                query_id=query_id,
                process_queue=process_queue
            ),
            agent_response_callback=agent_response_callback
        )

        self.runtime = InProcessRuntime()
        self.runtime.start()

        flow_event(self.logger, stage="orquestrador", message="Executando invoke() do group chat.", query_id=query_id)
        orchestration_result = await self.group_chat_orchestration.invoke(
            task=query,
            runtime=self.runtime,
        )
        flow_event(self.logger, stage="orquestrador", message="invoke() concluído.", status="success", query_id=query_id)

        final_response_message = await orchestration_result.get()
        raw_content = final_response_message.content
        flow_event(self.logger, stage="raw_response", message=raw_content, query_id=query_id)

        response_dict = None

        def is_valid_response(text):
            return (text and
                    len(text.strip()) > 5 and  # Pelo menos 5 caracteres
                    text.lower() not in ['erro', 'error', 'n/a', 'null'])

        try:
            clean = raw_content.strip("`")
            
            if clean.startswith("```json"):
                clean = clean[len("```json"):].strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1]
            if clean.endswith("```"):
                clean = clean.rsplit("\n", 1)[0]

            match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                clean = match.group(0).strip()
                parsed_response = json.loads(clean)

                if isinstance(parsed_response, dict) and "response" in parsed_response:
                    response_dict = {
                        "response": parsed_response.get("response", "Erro ao processar a resposta do modelo."),
                        "sql": parsed_response.get("sql", "N/A"),
                        "plot": parsed_response.get("plot", "N/A")
                    }
                    if not is_valid_response(response_dict["response"]):
                        self.logger.warning("Resposta dentro do JSON nao é valida ou Util.")
                        if is_valid_response(raw_content):
                            response_dict["response"] = raw_content
                else:
                    if is_valid_response(raw_content):
                        response_dict = {
                            "response": raw_content,
                            "sql": "N/A",
                            "plot": "N/A"
                        }
        except json.JSONDecodeError as e:
            self.logger.warning(f"Resposta nao esta em formato JSON valido: {e}")
            try:
                raw_content = raw_content.replace('"', "")
                modified_content = raw_content.replace("'", '"')
                parsed_response = json.loads(modified_content)
                
                if isinstance(parsed_response, dict) and "response" in parsed_response:
                    response_dict = {
                        "response": parsed_response.get("response", "Ocorreu algum erro ao responder sua pergunta. Gostaria de ajuda em algo mais?"),
                        "sql": parsed_response.get("sql", "N/A"),
                        "plot": parsed_response.get("plot", "N/A")
                    }
                    
                    if not is_valid_response(response_dict["response"]):
                        self.logger.warning("Resposta dentro do JSON modificado nao é valida ou util.")
                        if is_valid_response(raw_content):
                            response_dict["response"] = raw_content
            except Exception as e2:
                self.logger.warning(f"Falha ao analisar resposta modificada: {e2}")

            if is_valid_response(raw_content) and response_dict is None:
                response_dict = {
                    "response": raw_content,
                    "sql": "N/A",
                    "plot": "N/A"
                }
        except Exception as e:
            if is_valid_response(raw_content):
                response_dict = {
                    "response": raw_content,
                    "sql": "N/A",
                    "plot": "N/A"
                }

        if response_dict is None:
            response_dict = {
                "response": raw_content,
                "sql": "N/A",
                "plot": "N/A"
            }

        flow_event(self.logger, stage="response", message=f"Resposta final preparada. sql={response_dict.get('sql', 'N/A')}", status="success", query_id=query_id)

        final_response_for_query_output = {
            k: str(v) if v is not None else "N/A" for k, v in response_dict.items()
        }

        return QueryOutput(**final_response_for_query_output)

    async def run(self, query, summary, query_id, process_queue) -> str:
        ntry = 0
        last_error = None

        while ntry < self._max_try:
            try:
                self.logger.info(f"Tentativa {ntry+1} de {self._max_try}")
                return await self._message(query, summary, query_id, process_queue)
            
            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"Falha ao responder {e} na tentativa {ntry+1}. Aguardando {self._time_out} sec."
                )
                flow_event(self.logger, stage="retry", message=f"Falha na tentativa {ntry+1}: {e}", status="warning", query_id=query_id)
                self.logger.debug(traceback.format_exc())
                ntry += 1
                await self.stop_runtime()
                time.sleep(self._time_out)

        
        self.logger.error(f"Erro ao processar a pergunta após {self._max_try} tentativas: {last_error}")
        flow_event(self.logger, stage="erro", message=f"Erro após {self._max_try} tentativas: {last_error}", status="error", query_id=query_id)
        return QueryOutput(
            response="Desculpe, ocorreu um erro ao processar sua pergunta. Gostaria de ajuda em algo mais?",
        )

    async def stop_runtime(self):
        await self.runtime.close()