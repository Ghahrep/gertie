# in db/models.py

from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

# Import the Base class from our session manager.
# All our ORM models will inherit from this class.
from .session import Base

class User(Base):
    """
    Represents a user in the database.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    # This creates the one-to-many relationship.
    # The 'portfolios' attribute on a User object will be a list of
    # all Portfolio objects associated with this user.
    # The 'back_populates' argument links it to the 'owner' attribute
    # in the Portfolio model.
    portfolios = relationship("Portfolio", back_populates="owner")

class Asset(Base):
    """
    Represents a master list of all tradable assets (stocks, ETFs, etc.).
    This prevents us from duplicating asset information.
    """
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True) # e.g., "Apple Inc."

class Portfolio(Base):
    """
    Represents a single investment portfolio owned by a user.
    A user can have multiple portfolios.
    """
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    
    # This is the foreign key that links each portfolio to a user.
    user_id = Column(Integer, ForeignKey("users.id"))

    # This creates the many-to-one relationship back to the User.
    owner = relationship("User", back_populates="portfolios")
    
    # This creates the one-to-many relationship to the holdings within this portfolio.
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")

class Holding(Base):
    """
    Represents a specific quantity of an asset within a portfolio.
    """
    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, index=True)
    shares = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=True)  # This line
    
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    asset_id = Column(Integer, ForeignKey("assets.id"))

    portfolio = relationship("Portfolio", back_populates="holdings")
    asset = relationship("Asset")

    @hybrid_property
    def ticker(self):
        return self.asset.ticker