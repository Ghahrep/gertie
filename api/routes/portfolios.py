# in api/routes/portfolios.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from api import schemas
from db import crud
from db.session import get_db
from api.routes.auth import get_current_user
from db.models import Portfolio, Holding, Asset

router = APIRouter()

def serialize_holding(holding: Holding) -> dict:
    """Custom serialization for holdings with ticker and purchase_price"""
    return {
        "id": holding.id,
        "ticker": holding.asset.ticker if holding.asset else "",
        "shares": holding.shares,
        "purchase_price": holding.purchase_price,
        "asset_id": holding.asset_id,
    }

def serialize_portfolio(portfolio: Portfolio) -> dict:
    """Custom serialization for portfolios with properly serialized holdings"""
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "user_id": portfolio.user_id,
        "holdings": [serialize_holding(holding) for holding in portfolio.holdings],
    }

@router.post("/portfolios/", response_model=schemas.Portfolio, tags=["Portfolios"])
def create_new_portfolio(
    portfolio: schemas.PortfolioCreate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Create a new, empty portfolio for the current user."""
    return crud.create_portfolio(db=db, portfolio=portfolio, user_id=current_user.id)

@router.get("/portfolios/", tags=["Portfolios"])
def read_user_portfolios(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Retrieve all portfolios for the current user with properly serialized holdings."""
    # Get portfolios with holdings and assets joined
    portfolios = db.query(Portfolio).filter(
        Portfolio.user_id == current_user.id
    ).all()
    
    # Custom serialization to include ticker and purchase_price
    serialized_portfolios = []
    for portfolio in portfolios:
        # Ensure holdings are loaded with their assets
        db.refresh(portfolio)
        for holding in portfolio.holdings:
            db.refresh(holding)
            if holding.asset:
                db.refresh(holding.asset)
        
        serialized_portfolios.append(serialize_portfolio(portfolio))
    
    return serialized_portfolios

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
    holdings: List[schemas.HoldingCreate],
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Add a list of new asset holdings to a specific portfolio."""
    return crud.add_holdings_to_portfolio_bulk(db=db, portfolio_id=portfolio_id, holdings=holdings)