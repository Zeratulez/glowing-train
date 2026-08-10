from pydantic import BaseModel, computed_field

from src.core.settings import settings


class Test_Schema(BaseModel):
    __test__ = False
    text: str

class Cookies(BaseModel):
    session_id: str

class Metrics(BaseModel):
    response: str | Test_Schema | None = None
    latency: float
    input_tokens: int
    output_tokens: int
    # cached_tokens: int

    # @computed_field
    # @property
    # def cost(self) -> float:
        # missed_tokens = self.input_tokens - self.cached_tokens
        # return (((settings.INPUT_CACHE_HIT * self.cached_tokens) + (settings.INPUT_CACHE_MISS * missed_tokens) 
                 # + (settings.OUTPUT_TOKENS * self.output_tokens)) / 1_000_000.0)


class SearchArgs(BaseModel):
    query: str

class CalcArgs(BaseModel):
    expression: str