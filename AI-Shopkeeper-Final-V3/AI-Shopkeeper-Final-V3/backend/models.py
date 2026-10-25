from pydantic import BaseModel, Field


class InventoryUpdate(BaseModel):
    product_id: str
    current_stock: float = Field(ge=0)
    incoming_stock: float = Field(ge=0)
    safety_stock: float = Field(ge=0)


class ForecastRequest(BaseModel):
    product_id: str
    horizon: int = Field(default=7, ge=1, le=30)
    retrain: bool = False          # force re-training even if a cached model is valid
    tune: bool = False             # small hyper-parameter search (slower)
    lead_time_days: int = Field(default=2, ge=0, le=60)
    service_level: float = Field(default=0.95, gt=0.5, lt=0.999)
