# in api/routes/portfolios.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from api import schemas
from db import crud
from db.session import get_db
from api.routes.auth import get_current_user

router = APIRouter()

@router.post("/portfolios/", response_model=schemas.Portfolio, tags=["Portfolios"])
def create_new_portfolio(
    portfolio: schemas.PortfolioCreate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Create a new, empty portfolio for the current user."""
    return crud.create_portfolio(db=db, portfolio=portfolio, user_id=current_user.id)

@router.get("/portfolios/", response_model=List[schemas.Portfolio], tags=["Portfolios"])
def read_user_portfolios(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Retrieve all portfolios for the current user."""
    return crud.get_user_portfolios(db=db, user_id=current_user.id)

@router.post("/portfolios/{portfolio_id}/holdings/", response_model=schemas.Holding, tags=["Portfolios"])
def add_new_holding(
    portfolio_id: int,
    holding: schemas.HoldingCreate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Add a new asset holding to a specific portfolio."""
    # You might add a check here to ensure the user owns the portfolio
    return crud.add_holding_to_portfolio(db=db, portfolio_id=portfolio_id, holding=holding)

@router.post("/portfolios/{portfolio_id}/holdings/bulk-add", response_model=List[schemas.Holding], tags=["Portfolios"])
def add_new_holdings_bulk(
    portfolio_id: int,
    holdings: List[schemas.HoldingCreate], # The body is now a list of holdings
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Add a list of new asset holdings to a specific portfolio."""
    # Here you could add logic to verify the current_user owns the portfolio_id
    return crud.add_holdings_to_portfolio_bulk(db=db, portfolio_id=portfolio_id, holdings=holdings)
