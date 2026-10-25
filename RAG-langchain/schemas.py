from pydantic import BaseModel

#Pydantic models for request/response validation

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]