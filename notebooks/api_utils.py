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


class NDTChatGPTConnector:
    def __init__(self, provider_url: str, auth_token: str):
        self._server = provider_url
        self._auth = auth_token
        self._session = requests.Session()

    def send(self, context: List[Dict[str, Any]]) -> ResponseSchema:
        response = self._session.post(os.path.join(self._server, "chatgpt"), json=context,
                                      headers={"Authorization": f"Bearer {self._auth}"})
        response.raise_for_status()
        json_response = response.json()
        return ResponseSchema(**json_response)

    def __del__(self):
        self._session.close()
        