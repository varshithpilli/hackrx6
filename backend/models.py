from pydantic import BaseModel
from typing import List, Union

# Request Model
class QueryRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

# Response Model
class QueryResponse(BaseModel):
    answers: List[str]