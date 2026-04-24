from pydantic import BaseModel, Field
from enum import Enum, auto


class Type(Enum):
    string = auto()
    number = auto()
    integer = auto()
    boolean = auto()


class Parameter(BaseModel):
    name: str = Field(min_length=1)
    type: Type


class Function(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: list[Parameter] = Field(default_factory=list)


class Prompt(BaseModel):
    prompt: str = Field(min_length=1)
