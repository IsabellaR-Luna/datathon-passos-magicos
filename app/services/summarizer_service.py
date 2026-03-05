from openai import AzureOpenAI
from texttosql.domain.interfaces.summarizer_interface import SummarizerInterface
import requests
from shared.config.settings import settings

class SummarizerService(SummarizerInterface):
    def __init__(self):
        pass

    def summarize(self, text) -> str:
        prompt = f"""
    Resuma a seguinte conversa entre usuario e bot em poucas frases,
    destacando os pontos principais:
    {text}
    """
    
    
        endpoint = settings.ENDPOINT
        model_name = settings.GPTMODEL
        deployment = settings.GPTMODEL

        subscription_key = settings.APIKEY
        api_version = settings.APIVERSION

        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente que resume conversas."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=200,
            temperature=0.0,
            model=deployment
        )
        
        data = response
        response_text = data.choices[0].message.content
        print(f" RESUMO ==========={response_text}=============")
        return response_text
        