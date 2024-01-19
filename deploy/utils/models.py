from typing import Dict
from pydantic import BaseModel, Field


class UserInput(BaseModel):
    # sentence: str = Field(min_length=1, 
    #                       max_length=510)
    sentence: str

class DataOutput(BaseModel):
    response: Dict
