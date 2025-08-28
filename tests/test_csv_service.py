# test_csv_service.py - Run this to test your CSV service independently

import sys
import os

# Add your project root to Python path if needed
# sys.path.append('/path/to/your/project')

def test_csv_service():
    """Test the CSV service with your exact data"""
    
    try:
        # Import the service
        from api.routes.csv_import import EnhancedCSVService
        
        # Create service instance
        csv_service = EnhancedCSVService()
        
        # Your exact CSV data from the screenshot
        test_csv = """ticker,shares,purchase_price,purchase_date,name
ICE,100,150,1/15/2023,Intercontinental Exchange
GOOGL,50,2500,2/1/2023,Alphabet Inc.
TSLA,25,200,3/10/2023,Tesla Inc.
MSFT,75,300,1/20/2023,Microsoft Corporation
AVGO,30,450,2/15/2023,Broadcom
HOOD,100,40,3/10/2025,RobinHood"""
        
        print("🔍 Testing CSV Service...")
        print("=" * 50)
        
        # Test 1: Parse CSV
        print("1. Testing CSV parsing...")
        try:
            parsed_data = csv_service.parse_holdings_csv(test_csv)
            print(f"   ✅ Parsed {len(parsed_data)} rows")
            print(f"   📊 First row: {parsed_data[0] if parsed_data else 'None'}")
        except Exception as e:
            print(f"   ❌ Parse failed: {e}")
            return False
        
        # Test 2: Date parsing specifically
        print("\n2. Testing date parsing...")
        test_dates = ["1/15/2023", "2/1/2023", "3/10/2023"]
        for date_str in test_dates:
            parsed_date = csv_service.parse_date_flexible(date_str)
            print(f"   📅 '{date_str}' → '{parsed_date}'")
        
        # Test 3: Validation
        print("\n3. Testing validation...")
        try:
            validation_result = csv_service.validate_holdings_data(parsed_data)
            print(f"   ✅ Validation complete")
            print(f"   📈 Valid: {validation_result.is_valid}")
            print(f"   📊 Valid rows: {validation_result.valid_rows}")
            print(f"   ⚠️  Warnings: {len(validation_result.warnings)}")
            print(f"   ❌ Errors: {len(validation_result.errors)}")
            
            if validation_result.errors:
                print("   🔍 Errors details:")
                for error in validation_result.errors[:3]:
                    print(f"      • {error}")
            
            if validation_result.warnings:
                print("   🔍 Warnings details:")
                for warning in validation_result.warnings[:3]:
                    print(f"      • {warning}")
                    
            return validation_result.is_valid
            
        except Exception as e:
            print(f"   ❌ Validation failed: {e}")
            return False
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure csv_import.py is in the right location and has no syntax errors")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_api_endpoint():
    """Test if you can reach the CSV health endpoint"""
    
    try:
        import requests
        
        # Test your local API
        base_url = "http://localhost:8000"  # Adjust port if needed
        
        endpoints_to_test = [
            f"{base_url}/api/v1/csv/health",
            f"{base_url}/api/health",
            f"{base_url}/debug/csv-status"
        ]
        
        print("\n🌐 Testing API endpoints...")
        print("=" * 50)
        
        for endpoint in endpoints_to_test:
            try:
                response = requests.get(endpoint, timeout=5)
                print(f"   📡 {endpoint}: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"      📄 Response: {str(data)[:100]}...")
            except Exception as e:
                print(f"   📡 {endpoint}: ❌ {str(e)[:50]}...")
    
    except ImportError:
        print("   📦 requests not available for API testing")

if __name__ == "__main__":
    print("🧪 CSV Service Test Suite")
    print("=" * 50)
    
    # Test 1: Service functionality
    service_works = test_csv_service()
    
    # Test 2: API endpoints (if possible)
    test_api_endpoint()
    
    print("\n" + "=" * 50)
    if service_works:
        print("✅ CSV Service is working correctly!")
        print("Issue might be in routing or frontend integration.")
    else:
        print("❌ CSV Service has issues that need to be fixed.")
    
    print("\n🔧 Next steps:")
    print("1. If service works: Check FastAPI route registration")
    print("2. If service fails: Fix the CSV parsing/validation logic")
    print("3. Check frontend is calling the right endpoint")
    print("4. Verify file upload is reaching the backend")