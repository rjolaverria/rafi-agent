from dataclasses import dataclass
from openai import OpenAI


@dataclass
class Model:
    name: str
    client: OpenAI
