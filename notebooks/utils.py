from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (BaseMessage, ChatResult, AIMessage,
                              ChatGeneration, HumanMessage, SystemMessage,
                              ChatMessage, FunctionMessage)

from langchain.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain.utils import get_from_dict_or_env, get_pydantic_field_names

# import numpy as np
import dataclasses
from typing import Any, List, Dict, Union, Optional
# from typing import (
#     Callable,
#     Literal,
#     Optional,
#     Sequence,
#     Set,
#     Tuple,
# )

import requests
import os


@dataclasses.dataclass
class ChatGPTEntry:
    '''
    Зачем тебе читать эту документацию? 
    Лучше подписывайся на канал datafeeling в телеграм!
    '''
    role: str
    content: str

@dataclasses.dataclass
class ResponseSchema:
    '''
    Зачем тебе читать эту документацию? 
    Лучше подписывайся на канал datafeeling в телеграм!
    '''
    id : str
    object: str
    created: int
    model: str
    choices: Union[ChatGPTEntry, dict]
    usage: dict
    prompt_tokens: int
    completion_tokens: int
    available_tokens: int
    
    # def __post_init__(self):
        # self.choices = ChatGPTEntry(**self.choices[0])

class Struct(dict):
    def __init__(self, **entries):
        self.__dict__.update(entries)
        
    def __repr__(self):
        return str(self.__dict__)
             
class completions:
    '''
    Класс ChatCompletion по аналогии с одноименным классом из библиотеки openai
    '''
    
    _server = "https://api.neuraldeep.tech/" 
    _session = requests.Session()
    course_api_key = None

    def __init__(self, provider_url: str = "https://api.neuraldeep.tech/", **kwargs):
        self._server = provider_url
        self._session = requests.Session()

    @classmethod 
    def create(cls, messages: List[Dict[str, Any]],
               model="gpt-3.5-turbo",
               course_api_key: str = 'course_api_key',
               **kwargs) -> ResponseSchema:

        if cls.course_api_key is None:
            cls.course_api_key = course_api_key
        assert cls.course_api_key != 'course_api_key', 'Для генерации требуется ввести токен'
        
        messages = {'messages' : messages}
        messages.update(kwargs)

        cls._auth = cls.course_api_key
        response = cls._session.post(os.path.join(cls._server, "chatgpt"), json=messages,
                                      headers={"Authorization": f"Bearer {cls._auth}"})

        response.raise_for_status()
        json_response = response.json()

        # print(json_response)
        final_response = {}
        for k, v in json_response['raw_openai_response'].items():
            final_response[k] = v

        final_response['available_tokens'] = json_response['available_tokens'] 
        final_response['completion_tokens'] = json_response['completion_tokens'] 
        final_response['prompt_tokens'] = json_response['prompt_tokens']  

        # Меняем структуру словаря для вызова через точку (.)
        final_response['choices'][0]['message'] = Struct(**final_response['choices'][0]['message'])
        final_response['choices'] = [Struct(**final_response['choices'][0])]
   
        return ResponseSchema(**final_response)

    def __del__(self):
        self._session.close()
            
class chat:
    completions = completions
    
class OpenAI:
    def __init__(self, course_api_key="api"):
        self.course_api_key=course_api_key
        self.chat = chat
        self.chat.completions.course_api_key = course_api_key  

        
class ChatOpenAI(BaseChatModel):
        
    '''
    Класс ChatOpenAI по аналогии с одноименным классом из библиотеки openai
    '''
    
    course_api_key: str
    provider_url: str = "https://api.neuraldeep.tech/"
    client: completions = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = completions(provider_url=self.provider_url, **kwargs) #course_api_key=kwargs["course_api_key"])

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
            message=AIMessage(content=response.choices[0].message.content),
            generation_info=dict()
        )
        generations.append(gen)
        llm_output = {"token_usage": response.completion_tokens, "token_available": response.available_tokens}
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "Datafeeling ChatGPT proxy"
        
    
class Embedding:
    
    '''
    Класс Embedding по аналогии с одноименным классом из библиотеки openai
    '''

    # _session = requests.Session()
    model="text-embedding-ada-002"
    
    def __init__(self,
                 course_api_key : str,
                 provider_url: str = "https://api.neuraldeep.tech/", 
                 **kwargs):
        self._server = provider_url
        self._session = requests.Session()
        self.course_api_key = course_api_key
         # cls._auth = course_api_key
        
    # @classmethod 
    def create(self, input: str, **kwargs) -> ResponseSchema:
        
        # cls._auth = self.course_api_key
        # print(input)
        messange = {'str_to_vec' : input}
        response = self._session.post(os.path.join(self._server, "embeddings"), json=messange,
                                      headers={"Authorization": f"Bearer {self.course_api_key}"})
        
        response.raise_for_status()
        json_response = response.json()
        
        # print(json_response)
        # final_response = {}
        # for k,v in json_response['raw_openai_response'].items():
        #     final_response[k] = v

        # final_response['embedding'] = json_response['embedding'] 
        # final_response['available_tokens'] = json_response['available_tokens'] 
        # final_response['prompt_tokens'] = json_response['prompt_tokens']  
        
        return json_response['raw_openai_response']

from langchain.embeddings import OpenAIEmbeddings

class OpenAIEmbeddings(OpenAIEmbeddings):
    
    course_api_key : str
    provider_url: str = "https://api.neuraldeep.tech/"
    # client: Embedding 
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = Embedding(provider_url=self.provider_url, **kwargs)

    
    
    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["course_api_key"] = get_from_dict_or_env(
            values, "course_api_key", "COURSE_API_KEY"
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values,
            "openai_api_type",
            "OPENAI_API_TYPE",
            default="",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        if values["openai_api_type"] in ("azure", "azure_ad", "azuread"):
            default_api_version = "2022-12-01"
            # Azure OpenAI embedding models allow a maximum of 16 texts
            # at a time in each batch
            # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#embeddings
            default_chunk_size = 16
        else:
            default_api_version = ""
            default_chunk_size = 1000
        values["openai_api_version"] = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
            default=default_api_version,
        )
        values["openai_organization"] = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        if "chunk_size" not in values:
            values["chunk_size"] = default_chunk_size

        return values
    
# messages = [{"role": "user", "content": prompt}]
# response = ChatCompletion().create(course_api_key=course_api_key,
#                                    model = model,
#                                    messages = messages,
#                                    temperature=0)

# print(response)
    
# llm_chat = ChatOpenAI(course_api_key="вставь сюда токен курса")
# res = llm_chat(messages=[HumanMessage(content="Translate this sentence")])
# print(res)
