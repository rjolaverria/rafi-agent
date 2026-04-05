from dataclasses import dataclass
from openai import AsyncOpenAI


@dataclass
class Model:
    name: str
    client: AsyncOpenAI
