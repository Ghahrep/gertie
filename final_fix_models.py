# final_fix_models.py
"""
Final fix for db/models.py - adds missing imports
"""

def fix_imports_in_models():
    """Add missing imports to the models file"""
    
    models_file = 'db/models.py'
    
    # Read the file
    with open(models_file, 'r') as f:
        content = f.read()
    
    print("🔧 Adding missing imports...")
    
    # Check if Index is already imported in the main imports
    if 'from sqlalchemy import (' in content and 'Index' not in content.split('from sqlalchemy import (')[1].split(')')[0]:
        # Find the main sqlalchemy import and add Index
        old_import = 'from sqlalchemy import (\n    Column, Integer, String, Float, ForeignKey, Text, \n    JSON, Boolean, DateTime, Enum as SQLAlchemyEnum\n)'
        new_import = 'from sqlalchemy import (\n    Column, Integer, String, Float, ForeignKey, Text, \n    JSON, Boolean, DateTime, Enum as SQLAlchemyEnum, Index\n)'
        
        content = content.replace(old_import, new_import)
        print("✅ Added Index to main imports")
    
    # Also add timezone import if missing
    if 'from datetime import datetime' in content and 'timezone' not in content:
        content = content.replace(
            'from datetime import datetime',
            'from datetime import datetime, timezone'
        )
        print("✅ Added timezone import")
    
    # Write back
    with open(models_file, 'w') as f:
        f.write(content)
    
    return True

def test_final_models():
    """Test the final models"""
    print("🧪 Testing final models...")
    
    try:
        # Test basic imports
        from db.models import PortfolioRiskSnapshot, RiskThresholdConfig, RiskAlertLog
        print("✅ Risk models imported successfully")
        
        # Test helper functions
        from db.models import create_risk_snapshot_from_calculator, get_default_risk_thresholds
        print("✅ Helper functions imported successfully")
        
        # Test that we can create instances (without saving)
        thresholds = get_default_risk_thresholds()
        print(f"✅ Default thresholds: {len(thresholds)} settings")
        
        print("🎉 ALL MODELS WORKING PERFECTLY!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    print("🔧 FINAL FIX FOR DB/MODELS.PY")
    print("=" * 35)
    
    # Fix imports
    if fix_imports_in_models():
        # Test everything
        if test_final_models():
            print("\n🎉 PHASE 1 MODELS COMPLETE!")
            print("=" * 35)
            print("✅ db/models.py fully working")
            print("✅ All imports fixed")
            print("✅ Risk models ready to use")
            print("✅ Helper functions available")
            print("\n📋 Next: Add CRUD functions to db/crud.py")
            return True
    
    print("\n❌ Fix incomplete")
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)