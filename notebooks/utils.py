import functools
import inspect
from typing import Callable, cast, Literal, List, Union, Optional, Dict
import httpx
import pydantic
from openai import OpenAI, Stream, APIResponseValidationError
from openai._base_client import make_request_options
from openai._models import validate_type, construct_type, BaseModel
from openai._resource import SyncAPIResource
from openai._types import ResponseT, ModelBuilderProtocol, NotGiven, NOT_GIVEN, Headers, Query, Body
from openai._utils import maybe_transform, required_args
from openai.resources.chat import Completions as ChatCompletions
from openai.resources import Completions
from openai.types import CreateEmbeddingResponse, Completion, Embedding
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, completion_create_params, \
    ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam, ChatCompletionChunk
from langchain.chat_models import ChatOpenAI as GPT
from langchain.embeddings import OpenAIEmbeddings as OpenAIEmbeds


class ChatGPTEntry(BaseModel):
    role: str
    content: str


class ResponseSchema(BaseModel):
    response: ChatGPTEntry
    prompt_tokens: int
    completion_tokens: int
    available_tokens: int
    raw_openai_response: Union[ChatCompletion, Completion, None]  = None


def chat_completion_overload(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # trick to get openai schema from NDT custom schema
        # here is not ChatCompletion, here NDT schema but wrong casted ChatCompletion to inside openai lib
        #print(args, kwargs)
        result: ChatCompletion | Stream = func(*args, **kwargs)
        if isinstance(result, Stream):
            return result

        #print(result)
        ndt_response = ResponseSchema(**result.model_dump(exclude_unset=True, exclude_defaults=True))
        #print(ndt_response.available_tokens)
        return ndt_response.raw_openai_response

    return wrapper


class NDTChatCompletions(ChatCompletions):
    
    @required_args(["messages", "model"], ["messages", "model", "stream"])
    def create(
            self,
            *,
            messages: List[ChatCompletionMessageParam],
            model: Union[
                str,
                Literal[
                    "gpt-4-1106-preview",
                    "gpt-4-vision-preview",
                    "gpt-4",
                    "gpt-4-0314",
                    "gpt-4-0613",
                    "gpt-4-32k",
                    "gpt-4-32k-0314",
                    "gpt-4-32k-0613",
                    "gpt-3.5-turbo-1106",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-16k",
                    "gpt-3.5-turbo-0301",
                    "gpt-3.5-turbo-0613",
                    "gpt-3.5-turbo-16k-0613",
                ],
            ],
            frequency_penalty: Optional[float] = NOT_GIVEN,
            function_call: completion_create_params.FunctionCall = NOT_GIVEN,
            functions: List[completion_create_params.Function] = NOT_GIVEN,
            logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
            max_tokens: Optional[int] = NOT_GIVEN,
            n: Optional[int] = NOT_GIVEN,
            presence_penalty: Optional[float] = NOT_GIVEN,
            response_format: completion_create_params.ResponseFormat = NOT_GIVEN,
            seed: Optional[int] = NOT_GIVEN,
            stop: Union[Optional[str], List[str]] = NOT_GIVEN,
            stream: Optional[Literal[False]] = NOT_GIVEN,
            temperature: Optional[float] = NOT_GIVEN,
            tool_choice: ChatCompletionToolChoiceOptionParam = NOT_GIVEN,
            tools: List[ChatCompletionToolParam] = NOT_GIVEN,
            top_p: Optional[float] = NOT_GIVEN,
            user: str = NOT_GIVEN,
            # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
            # The extra values given here take precedence over values defined on the client or passed to this method.
            extra_headers: Headers = None,
            extra_query: Query = None,
            extra_body: Body = None,
            timeout: float = NOT_GIVEN,
    ) -> ChatCompletion:
        result: ResponseSchema = self._post(
            "/chat/completions",
            body=maybe_transform(
                {
                    "messages": messages,
                    "model": model,
                    "frequency_penalty": frequency_penalty,
                    "function_call": function_call,
                    "functions": functions,
                    "logit_bias": logit_bias,
                    "max_tokens": max_tokens,
                    "n": n,
                    "presence_penalty": presence_penalty,
                    "response_format": response_format,
                    "seed": seed,
                    "stop": stop,
                    "stream": stream,
                    "temperature": temperature,
                    "tool_choice": tool_choice,
                    "tools": tools,
                    "top_p": top_p,
                    "user": user,
                },
                completion_create_params.CompletionCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ResponseSchema,
            stream=stream or False,
            stream_cls=Stream[ChatCompletionChunk],
        )

        #print(result)
        return result.raw_openai_response


class NDTCompletions(Completions):
    
    @required_args(["model", "prompt"], ["model", "prompt", "stream"])
    def create(
        self,
        *,
        model: Union[
            str,
            Literal[
                "babbage-002",
                "davinci-002",
                "gpt-3.5-turbo-instruct",
                "text-davinci-003",
                "text-davinci-002",
                "text-davinci-001",
                "code-davinci-002",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001",
            ],
        ],
        prompt: Union[str, List[str], List[int], List[List[int]], None],
        best_of: Optional[int] = NOT_GIVEN,
        echo: Optional[bool] = NOT_GIVEN,
        frequency_penalty: Optional[float] = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] = NOT_GIVEN,
        logprobs: Optional[int] = NOT_GIVEN,
        max_tokens: Optional[int] = NOT_GIVEN,
        n: Optional[int] = NOT_GIVEN,
        presence_penalty: Optional[float] = NOT_GIVEN,
        seed: Optional[int] = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] = NOT_GIVEN,
        stream: Optional[Literal[False]] = NOT_GIVEN,
        suffix: Optional[str] = NOT_GIVEN,
        temperature: Optional[float] = NOT_GIVEN,
        top_p: Optional[float] = NOT_GIVEN,
        user: str = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers = None,
        extra_query: Query = None,
        extra_body: Body = None,
        timeout: float = NOT_GIVEN,
    ) -> Completion:
        result: ResponseSchema = self._post(
            "/completions",
            body=maybe_transform(
                {
                    "model": model,
                    "prompt": prompt,
                    "best_of": best_of,
                    "echo": echo,
                    "frequency_penalty": frequency_penalty,
                    "logit_bias": logit_bias,
                    "logprobs": logprobs,
                    "max_tokens": max_tokens,
                    "n": n,
                    "presence_penalty": presence_penalty,
                    "seed": seed,
                    "stop": stop,
                    "stream": stream,
                    "suffix": suffix,
                    "temperature": temperature,
                    "top_p": top_p,
                    "user": user,
                },
                completion_create_params.CompletionCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=ResponseSchema,
            stream=stream or False,
            stream_cls=Stream[Completion],
        )
        
        #print(result)
        import time
        time.sleep(5)
        return result.raw_openai_response


    
class NDTChat(SyncAPIResource):
    completions: NDTChatCompletions

    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self.completions = NDTChatCompletions(client)


class EmbeddingResponseSchema(BaseModel):
    data: list[Embedding]
    prompt_tokens: int
    available_tokens: int
    raw_openai_response: CreateEmbeddingResponse = None


def embeddings_overload(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # trick to get openai schema from NDT custom schema
        # here is not CreateEmbeddingResponse, here NDT schema but wrong casted CreateEmbeddingResponse to inside openai lib
        result: CreateEmbeddingResponse = func(*args, **kwargs)
        ndt_response = EmbeddingResponseSchema(**result.model_dump(exclude_unset=True, exclude_defaults=True))
        #print(ndt_response.available_tokens)
        return ndt_response.raw_openai_response

    return wrapper


class NDTOpenAI(OpenAI):
    chat: NDTChat
    completions: NDTCompletions
    server_url: str = "https://api.neuraldeep.tech/"

    def __init__(self, api_key, **kwargs):
        super().__init__(api_key=api_key, base_url=self.server_url, **kwargs)
        self.embeddings.create = embeddings_overload(self.embeddings.create)
        self.chat = NDTChat(self)
        self.completions = NDTCompletions(self)
        
        
class ChatOpenAI(GPT):
        
    '''
    Класс ChatOpenAI по аналогии с одноименным классом из библиотеки langchain
    '''
    
    openai_api_key: str = 'api_key'
    
    def __init__(self, course_api_key, **kwargs):
        super().__init__(client = NDTOpenAI(api_key=course_api_key).chat.completions, **kwargs)


class OpenAIEmbeddings(OpenAIEmbeds):
    
    '''
    Класс OpenAIEmbeddings по аналогии с одноименным классом из библиотеки langchain
    '''
    
    openai_api_key: str = 'api_key'
    
    
    def __init__(self, course_api_key, **kwargs):
        super().__init__(client = NDTOpenAI(api_key=course_api_key).embeddings, **kwargs)
