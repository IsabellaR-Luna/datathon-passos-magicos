from typing import Any, List, Optional, AsyncGenerator, cast, TypeVar, Callable
import json
import sys
import logging
from uuid import uuid4
from openai.types.chat.chat_completion import ChatCompletion, Choice
# import openai
# from openai import FunctionCall
# from openai import ChatCompletionChunk, ChoiceDeltaFunctionCall, ChoiceDeltaToolCall
# from openai import ChatCompletionMessageToolCall
# from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
# from semantic_kernel.contents.chat_message_content import ChatMessageContent
# from semantic_kernel.connectors.ai.completion_usage import CompletionUsage
# from semantic_kernel.contents.function_call_content import FunctionCallcontent
# from semantic_kernel.connectors.ai.function_calling_utils import update_settings_from_function_call_configuration
# from semantic_kernel.contents.chat_history import AuthorRole, ChatHistory
# from semantic_kernel.contents.text_content import TextContent
# from semantic_kernel.exceptions import ServiceInvalidExecutionSettingsError, ServiceInvalidResponseError
# from semantic_kernel.contents.utils.finish_reason import FinishReason
# from semantic_kernel.kernel import Kernel
# from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings
# from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
# from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
# from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior, FunctionChoiceType
# from semantic_kernel.connectors.ai.function_call_choice_configuration import FunctionCallChoiceConfiguration
# from semantic_kernel.contents.function_result_content import FunctionResultContent
# from openai import AsyncAzureOpenAI

# OpenAI (SDK novo – Azure)
from openai import AsyncAzureOpenAI

# Semantic Kernel – núcleo
from semantic_kernel.kernel import Kernel

# Chat / histórico
from semantic_kernel.contents.chat_history import ChatHistory, AuthorRole
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.text_content import TextContent

# Function calling
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent

# Finalização / status
from semantic_kernel.contents.utils.finish_reason import FinishReason

# Base de clientes e métricas
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.completion_usage import CompletionUsage

# Execution settings
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)

# Comportamento de escolha de funções
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
    FunctionChoiceType,
)

from semantic_kernel.connectors.ai.function_call_choice_configuration import (
    FunctionCallChoiceConfiguration,
)

from semantic_kernel.connectors.ai.function_calling_utils import (
    update_settings_from_function_call_configuration,
)

# Exceções
from semantic_kernel.exceptions import (
    ServiceInvalidExecutionSettingsError,
    ServiceInvalidResponseError,
)


if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

logger: logging.Logger = logging.getLogger(__name__)
TChatMessageContent = TypeVar("TChatMessageContent", ChatMessageContent, StreamingChatMessageContent)

class AzureOpenAIChatCompletionClient(ChatCompletionClientBase):
    endpoint: str
    api_key: str
    api_version: str
    _client: Optional[AsyncAzureOpenAI] = None
    _tools: Optional[list] = None

    def __init__(
        self,
        service_id: str,
        deployment_name: str,
        endpoint: str,
        api_key: str,
        api_version: str,
        tools: Optional[list] = None
    ):
        super().__init__(
            ai_model_id=deployment_name,
            service_id=service_id,
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            tools=tools
        )
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.tools = tools

    @property
    def tools(self) -> Optional[list]:
        return self._tools

    @tools.setter
    def tools(self, value: Optional[list]):
        self._tools = value

    @property
    def client(self) -> AsyncAzureOpenAI:
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
        return self._client

    def _get_metadata_from_chat_response(self, response: ChatCompletion) -> dict[str, Any]:
        """Get metadata from a chat response."""
        return {
            "id": response.id,
            "created": response.created,
            "system_fingerprint": response.system_fingerprint,
            "usage": CompletionUsage.from_openai(response.usage) if response.usage is not None else None,
        }

    def _get_metadata_from_chat_choice(self, choice: Choice) -> dict[str, Any]:
        """Get metadata from a chat choice."""
        return {
            "logprobs": getattr(choice, "logprobs", None),
        }

    def _get_tool_calls_from_chat_choice(self, choice: Choice) -> list[FunctionCallContent]:
        """Get tool calls from a chat choice."""
        content = choice.message
        if content and (tool_calls := getattr(content, "tool_calls", None)) is not None:
            return [
                FunctionCallContent(
                    id=tool.id,
                    index=getattr(tool, "index", None),
                    name=tool.function.name,
                    arguments=tool.function.arguments,
                )
                for tool in cast(list[ChatCompletionMessageToolCall] | list[ChoiceDeltaToolCall], tool_calls)
                if tool.function is not None
            ]
        # When you enable asynchronous content filtering in Azure OpenAI, you may receive empty deltas
        return []

    def _get_function_call_from_chat_choice(self, choice: Choice) -> list[FunctionCallContent]:
        """Get a function call from a chat choice."""
        content = choice.message
        if content and (function_call := getattr(content, "function_call", None)) is not None:
            function_call = cast(FunctionCall | ChoiceDeltaFunctionCall, function_call)
            return [
                FunctionCallContent(
                    id="legacy_function_call",
                    name=function_call.name,
                    arguments=function_call.arguments
                )
            ]
        return []

    def _create_chat_message_content(
        self,
        response: ChatCompletion,
        choice: Choice,
        response_metadata: dict[str, Any]
    ) -> "ChatMessageContent":
        """Create a chat message content object from a choice."""
        metadata = self._get_metadata_from_chat_choice(choice)
        metadata.update(response_metadata)

        items: list[Any] = self._get_tool_calls_from_chat_choice(choice)
        items.extend(self._get_function_call_from_chat_choice(choice))

        if choice.message.content:
            items.append(TextContent(text=choice.message.content))
        elif hasattr(choice.message, "refusal") and choice.message.refusal:
            items.append(TextContent(text=choice.message.refusal))

        content = ChatMessageContent(
            inner_content=response,
            ai_model_id=self.ai_model_id,
            metadata=metadata,
            role=AuthorRole(choice.message.role),
            items=items,
            finish_reason=(FinishReason(choice.finish_reason) if choice.finish_reason else None),
        )
        return self._add_tool_message_to_chat_message_content(content, choice)

    def _add_tool_message_to_chat_message_content(
        self,
        content: TChatMessageContent,
        choice: Choice,
    ) -> TChatMessageContent:
        if tool_message := self._get_tool_message_from_chat_choice(choice=choice):
            if not isinstance(tool_message, dict):
                # try to json, to ensure it is a dictionary
                try:
                    tool_message = json.loads(tool_message)
                except json.JSONDecodeError:
                    logger.warning("Tool message is not a dictionary, ignore context.")
                    return content
            function_call = FunctionCallContent(
                id=str(uuid4()),
                name="Azure-OnYourData",
                arguments=json.dumps({"query": tool_message.get("intent", [])}),
            )
            result = FunctionResultContent.from_function_call_content_and_result(
                result=tool_message["citations"], function_call_content=function_call
            )
            content.items.insert(0, function_call)
            content.items.insert(1, result)
        return content

    def _get_tool_message_from_chat_choice(self, choice: Choice) -> dict[str, Any] | None:
        """Get the tool message from a choice."""
        content = choice.message
        # When you enable asynchronous content filtering in Azure OpenAI, you may receive empty deltas
        if content and content.model_extra is not None:
            return content.model_extra.get("context", None)
        # openai allows extra content, so model_extra will be a dict, but we need to check anyway, but no way to test.
        return None  # pragma: no cover

    @override
    def get_prompt_execution_settings_class(self) -> type["PromptExecutionSettings"]:
        return OpenAIChatPromptExecutionSettings

    async def _inner_get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: Optional[Any] = None,
        kernel: Optional[Kernel] = None,
        **kwargs
    ) -> list[ChatMessageContent]:
        if not isinstance(settings, OpenAIChatPromptExecutionSettings):
            settings = self.get_prompt_execution_settings_from_settings(settings)
        assert isinstance(settings, OpenAIChatPromptExecutionSettings)  # nosec
        messages = [{"role": m.role, "content": m.content} for m in chat_history.messages]
        response = await self.__call_model(kwargs, messages, kernel)
        response_metadata = self._get_metadata_from_chat_response(response)
        final_response = [self._create_chat_message_content(response, choice, response_metadata) for choice in response.choices]

        for choice in response.choices:
            if getattr(choice.message, "tool_calls", None):
                for tool_call in choice.message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    if kernel is None:
                        continue
                    try:
                        function = None
                        all_plugins = getattr(kernel, "plugins", {})
                        if isinstance(all_plugins, dict):
                            for plugin_name, plugin_obj in all_plugins.items():
                                functions = getattr(plugin_obj, "functions", {})
                                if isinstance(functions, dict) and fn_name in functions:
                                    function = functions[fn_name]
                                    break
                        if function is None:
                            continue
                        result = await kernel.invoke(function, **fn_args)
                    except Exception as e:
                        continue
                    messages.append({
                        "role": "assistant",
                        "content": "Tool Call",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": fn_name,
                                    "arguments": json.dumps(fn_args)
                                }
                            }
                        ]
                    })
                    messages.append({
                        "role": "assistant",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result.value if hasattr(result, 'value') else str(result))
                    })
                    followup = await self.__call_model(kwargs, messages, kernel)
                    final_response = [self._create_chat_message_content(followup, c, response_metadata) for c in followup.choices]
        return final_response

    async def __call_model(self, kwargs, messages, kernel):
        available_plugins = getattr(kernel, "plugins", {})
        tool_choice = ""
        last_message = messages[-1] if messages else None
        if last_message:
            print(f"[CALL MODEL] Last message role: {last_message.get('role')}, content: {last_message.get('content')}")
        if isinstance(available_plugins, dict) and len(available_plugins) > 0:
            tool_choice = "auto"
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_empresa",
                        "description": "Pesquisa o nome de uma empresa na base de dados e retorna as informacdes dela, caso encontrada. Exemplo de usabilidade: get_empresa('Alupar')",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "nome_empresa": {
                                    "type": "string",
                                    "description": "Nome da empresa a ser pesquisada na base de dados."
                                }
                            },
                            "required": ["nome_empresa"]
                        }
                    }
                }
            ]
        else:
            tool_choice = None
            tools = None
        print(f"[CALL MODEL] tool_choice={tool_choice}, tools={tools}")
        allowed_params = {"max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"}
        api_kwargs = {key: value for key, value in kwargs.items() if key in allowed_params}
        if "temperature" not in api_kwargs:
            api_kwargs["temperature"] = 0.0
        if "max_tokens" not in api_kwargs:
            api_kwargs["max_tokens"] = 4096
        response = await self.client.chat.completions.create(
            model=self.ai_model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **api_kwargs
        )
        return response

    @override
    def _verify_function_choice_settings(self, settings: "PromptExecutionSettings") -> None:
        if not isinstance(settings, OpenAIChatPromptExecutionSettings):
            raise ServiceInvalidExecutionSettingsError("The settings must be an OpenAIChatPromptExecutionSettings.")
        if settings.number_of_responses is not None and settings.number_of_responses > 1:
            raise ServiceInvalidExecutionSettingsError(
                "Auto-invocation of tool calls may only be used with a OpenAIChatPromptExecutions.number_of_responses of 1."
            )

    @override
    def _update_function_choice_settings_callback(
        self,
    ) -> Callable[[FunctionCallChoiceConfiguration, PromptExecutionSettings, FunctionChoiceType], None]:
        return update_settings_from_function_call_configuration

    @override
    def _reset_function_choice_settings(self, settings: "PromptExecutionSettings") -> None:
        if hasattr(settings, "tool_choice"):
            settings.tool_choice = None
        if hasattr(settings, "tools"):
            settings.tools = None

    @override
    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: Optional[Any] = None,
        kernel: Optional[Kernel] = None,
        **kwargs
    ) -> List[ChatMessageContent]:
        result = await self._inner_get_chat_message_contents(chat_history, settings, kernel, **kwargs)
        if isinstance(result, list):
            return result
        return [result]

    @override
    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: Optional[Any] = None,
        kernel: Optional[Kernel] = None,
        **kwargs
    ) -> AsyncGenerator[ChatMessageContent, None]:
        messages = [{"role": m.role, "content": m.content} for m in chat_history.messages]
        allowed_params = {"max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"}
        api_kwargs = {key: value for key, value in kwargs.items() if key in allowed_params}
        api_kwargs.setdefault("temperature", 0.0)
        api_kwargs.setdefault("max_tokens", 4096)
        use_tools = bool(self.tools)
        buffer = ""
        try:
            stream_response = await self.client.chat.completions.create(
                model=self.ai_model_id,
                messages=messages,
                stream=True,
                tools=self.tools if use_tools else None,
                tool_choice="auto" if use_tools else None,
                **api_kwargs
            )
            async for chunk in stream_response:
                if not chunk.choices or not chunk.choices[0].delta:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if choice.finish_reason is not None:
                    if buffer.strip():
                        final_message = [
                            ChatMessageContent(
                                role=AuthorRole.ASSISTANT,
                                content=buffer
                            )
                        ]
                        yield final_message
                    buffer = ""
                token = getattr(delta, "content", "")
                if token == "" or token is None:
                    continue
                buffer += token
        except Exception as e:
            if buffer.strip():
                final_message = [ChatMessageContent(role=AuthorRole.ASSISTANT, content=buffer)]
                yield final_message
            yield [ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content=f"Erro durante streaming: {str(e)}"
            )]

    async def _tools_calling(self, tool_calls: list, messages: list, i_chamada: int = 0, **kwargs) -> list:
        message = []
        logger = logging.getLogger("ChatbotAPI")
        logger.info(f"[Tools Calling] Iniciando chamada de ferramentas. tool_calls={tool_calls}")
        response = None
        for tool_call in tool_calls:
            message.extend(self.__function_calling(message, tool_call))
            messages.extend(message)
            response = await self.__call_model(kwargs, messages)
            new_message = response.choices[0].message
            if hasattr(new_message, 'tool_calls') and new_message.tool_calls:
                return await self._tools_calling(new_message.tool_calls, messages, i_chamada=i_chamada, **kwargs)
        if response is None:
            final_response = await self.__call_model(kwargs, messages)
        else:
            final_response = response
        output_text = final_response.choices[0].message.content
        max_try = 3
        n_try = 0
        while output_text is None:
            final_response = await self.__call_model(kwargs, messages)
            output_text = final_response.choices[0].message.content
            n_try += 1
            if n_try >= max_try:
                output_text = '{"response": "Desculpe, nao consegui processar sua solicitacao.", "sql": "N/A", "plot": "N/A"}'
                break
        return [ChatMessageContent(role="assistant", content=output_text)]

    def __function_calling(self, message, tool_call):
        function_name = tool_call.function.name
        if not isinstance(message, list):
            if isinstance(message, str):
                message_text = message
                message = [{
                    "role": "assistant",
                    "content": function_name,
                    "tool_call_id": tool_call.id
                }]
                message.append({
                    "role": "assistant",
                    "tool_call_id": tool_call.id,
                    "content": f"tool_answer({message_text})"
                })
            else:
                return []
        args = json.loads(tool_call.function.arguments)
        result = eval(function_name)(**args)
        logger = logging.getLogger("ChatbotAPI")
        logger.info(f"[TOOL RAW RESULT] {repr(result)}")
        safe_result = json.dumps({"result": result}, ensure_ascii=False)
        message.append({
            "role": "assistant",
            "content": function_name,
            "tool_call_id": tool_call.id
        })
        message.append({
            "role": "assistant",
            "tool_call_id": tool_call.id,
            "content": f"tool_answer({safe_result})"
        })
        return message