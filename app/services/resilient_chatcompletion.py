import asyncio

from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent

class ResilientChatCompletionService(ChatCompletionClientBase):

    """
    Wrapper que tenta usar streaming e, em caso de falha, faz fallback para a
    chamada nao-streaming e devolve um unico chunk.
    """

    def __init__(self, inner_service):
        self._inner = inner_service

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _sanitize_chat_history(self, chat_history: ChatHistory) -> ChatHistory:
        """
        Ajustar mensagens de tool_calls e function calls alterando elas
        por mensagens de assistente.
        """
        for i, m in enumerate(chat_history.messages):
            finish_reason = getattr(m, "finish_reason", None)
            if finish_reason == "tool_calls":
                # cria mensagem nova e substitui
                new_message = ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content=m.content if m.content else "Tool Call.",
                    name=m.name
                )
                chat_history.messages[i] = new_message
            role = getattr(m, "role", None)
            if role == AuthorRole.TOOL:
                # substitui role da mensagem por assistant
                chat_history.messages[i].role = AuthorRole.ASSISTANT
        return chat_history

    async def get_chat_message_content(self, chat_history, settings, **kwargs):
        chat_history = self._sanitize_chat_history(chat_history)
        return await self._inner.get_chat_message_content(chat_history, settings, **kwargs)

    async def get_chat_message_contents(self, chat_history, settings, **kwargs):
        return await self._inner.get_chat_message_contents(chat_history, settings, **kwargs)

    async def get_streaming_chat_message_contents(self, chat_history, settings, **kwargs):
        try:
            async for messages in self._inner.get_streaming_chat_message_contents(chat_history, settings, **kwargs):
                yield messages
            return
        except Exception as e:
            print(f"Streaming falhou: {e}. Tentando fallback para nao-streaming.")
            
        last_exc = None
        for attempt in range(10):
            try:
                chat_history = self._sanitize_chat_history(chat_history)
                result = await self._inner.get_chat_message_contents(chat_history, settings, **kwargs)
                if result is None:
                    return
                if isinstance(result, list):
                    yield result
                else:
                    yield [result]
                return
            except Exception as e2:
                last_exc = e2
                print(f"Fallback nao-streaming falhou na tentativa {attempt+1}: {e2}.")
                await asyncio.sleep(0.5)
        if last_exc:
            raise last_exc