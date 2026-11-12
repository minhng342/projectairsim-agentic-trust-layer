"""The base of every plan-file model: frozen, no unknown fields, no NaN or infinity."""
from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
