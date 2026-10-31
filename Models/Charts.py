from datetime import datetime
from pydantic import BaseModel

class Chart(BaseModel):
    time: datetime
    value:float
