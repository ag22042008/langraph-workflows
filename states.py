# 1 thing is always state in langgraph a shared memory btw agents
#so now we are creating a graph so first we are creating states
import os
# typed dictonary 1 state method
from typing import TypedDict
#creating a blueprint most common
class State(TypedDict):
    tpoic:str
    summary:str
    score: int
#2 is pydantic model
# it is good at data validation and type checking at run time
#runtime
from pydantic import BaseModel,field_validator
class State(BaseModel):
    topic:str
    score:int
    summary:str=""
    @field_validator
    def score_positive(cls,v):#(score)
        if v<0:
            raise ValueError("score must be positive")


#3 python data classes
#standard python data class but it is used very rarely
from dataclasses import dataclass,field
@dataclass
class State:
    topic : str=""
    summary : str=""
    message: list=field(default_factory=list)
from langgraph.graph import MessagesState
class State(MessagesState):
    user_name:str
    language:str
