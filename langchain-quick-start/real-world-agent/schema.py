from pydantic import BaseModel

class AgentResponseFormat(BaseModel):
    summary:str
    confidence:float