from typing import List, Optional, Any, Dict

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, AIMessage, ChatGeneration, HumanMessage, SystemMessage, \
    ChatMessage, FunctionMessage
# from client import NDTChatGPTConnector, ResponseSchema


import dataclasses
from typing import Any, List, Dict, Union

import requests
import os


@dataclasses.dataclass
class ChatGPTEntry:
    role: str
    content: str

@dataclasses.dataclass
class ResponseSchema:
    response: Union[ChatGPTEntry, dict]
    prompt_tokens: int
    completion_tokens: int
    available_tokens: int

    def __post_init__(self):
        self.response = ChatGPTEntry(**self.response)


class ChatCompletion:
    
    
    '''
    Класс ChatCompletion по аналогии с одноименным классом из библиотеки openai
    '''
    
    def __init__(self, provider_url: str = "https://api.neuraldeep.tech/", **kwargs):
        self._server = provider_url
        self._session = requests.Session()
        
    def create(self, messages: List[Dict[str, Any]],
               model="gpt-3.5-turbo",
               course_api_key: str = 'course_api_key', **kwargs) -> ResponseSchema:
        
        assert course_api_key != 'course_api_key' and course_api_key is not None, 'Для генерации требуется ввести токен'
        
        messages = {'messages' : messages}
        messages.update(kwargs)
        
        self._auth = course_api_key
        response = self._session.post(os.path.join(self._server, "chatgpt"), json=messages,
                                      headers={"Authorization": f"Bearer {self._auth}"})
        response.raise_for_status()
        json_response = response.json()
        return ResponseSchema(**json_response)

    def __del__(self):
        self._session.close()
        
class ChatOpenAI(BaseChatModel):
        
    '''
    Класс ChatOpenAI по аналогии с одноименным классом из библиотеки openai
    '''
    
    course_api_key: str
    provider_url: str = "https://api.neuraldeep.tech/"
    client: ChatCompletion = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = ChatCompletion(provider_url=self.provider_url, **kwargs) #course_api_key=kwargs["course_api_key"])

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        message_dict: Dict[str, Any]
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs["function_call"]
                # If function call only, content is None not empty string
                if message_dict["content"] == "":
                    message_dict["content"] = None
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, FunctionMessage):
            message_dict = {
                "role": "function",
                "content": message.content,
                "name": message.name,
            }
        else:
            raise TypeError(f"Got unknown type {message}")
        return message_dict

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        response = self.client.create([self._convert_message_to_dict(m) for m in messages], course_api_key=self.course_api_key)
        return self._create_chat_result(response)

    def _create_chat_result(self, response: ResponseSchema) -> ChatResult:
        generations = []
        gen = ChatGeneration(
            message=AIMessage(content=response.response.content),
            generation_info=dict()
        )
        generations.append(gen)
        llm_output = {"token_usage": response.completion_tokens, "token_available": response.available_tokens}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "NDT ChatGPT proxy"

    
# messages = [{"role": "user", "content": prompt}]
# response = ChatCompletion().create(course_api_key=course_api_key,
#                                    model = model,
#                                    messages = messages,
#                                    temperature=0)

# print(response)
    
# llm_chat = ChatOpenAI(course_api_key="вставь сюда токен курса")
# res = llm_chat(messages=[HumanMessage(content="Translate this sentence")])
# print(res)
