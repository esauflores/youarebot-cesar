from uuid import UUID
from pydantic import BaseModel, Field, StrictStr


class IncomingMessage(BaseModel):
    text: StrictStr
    dialog_id: UUID
    id: UUID
    participant_index: int


class Prediction(BaseModel):
    id: UUID
    message_id: UUID
    dialog_id: UUID
    participant_index: int
    is_bot_probability: float = Field(ge=0.0, le=1.0)
