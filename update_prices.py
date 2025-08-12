# update_holdings_prices.py
from db.session import get_db
from sqlalchemy.orm import sessionmaker
from db.models import Holding, Asset
from db.session import engine

def update_holding_prices():
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # Get all holdings with their assets
        holdings = db.query(Holding).join(Asset).all()
        
        # Define realistic purchase prices for each ticker
        price_map = {
            "AAPL": 150.00,
            "GOOGL": 140.00, 
            "MSFT": 350.00,
            "AMZN": 120.00,
            "TSLA": 200.00,
            "NVDA": 450.00,
            "META": 300.00,
            # Add more as needed
        }
        
        updated_count = 0
        
        for holding in holdings:
            ticker = holding.asset.ticker
            if ticker in price_map:
                holding.purchase_price = price_map[ticker]
                updated_count += 1
                print(f"✅ Updated {ticker}: {holding.shares} shares @ ${price_map[ticker]}")
        
        db.commit()
        print(f"\n🎉 Successfully updated {updated_count} holdings with purchase prices!")
        
        # Show portfolio totals
        print("\n📊 Portfolio Summary:")
        total_value = 0
        for holding in holdings:
            if holding.purchase_price:
                value = holding.shares * holding.purchase_price
                total_value += value
                print(f"   {holding.asset.ticker}: {holding.shares} shares × ${holding.purchase_price} = ${value:,.2f}")
        
        print(f"\n💰 Total Portfolio Value: ${total_value:,.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    update_holding_prices()