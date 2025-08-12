# in api/schemas.py
from pydantic import BaseModel, computed_field
from typing import Optional, List # Ensure List is imported

# --- Token and User Schemas (Unchanged) ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    
    class Config:
        from_attributes = True

# --- Holding and Portfolio Schemas (REVISED) ---
class HoldingBase(BaseModel):
    ticker: str # The ticker is now a regular field
    shares: float

class HoldingCreate(HoldingBase):
    pass # No change needed here

class Holding(HoldingBase):
    id: int
    asset_id: int
    purchase_price: Optional[float] = None

    class Config:
        from_attributes = True

# --- Portfolio Schemas (Unchanged) ---
class PortfolioBase(BaseModel):
    name: str

class PortfolioCreate(PortfolioBase):
    pass

class Portfolio(PortfolioBase):
    id: int
    user_id: int
    holdings: List[Holding] = []

    class Config:
        from_attributes = True