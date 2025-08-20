import pytest
import importlib
import inspect

def test_check_existing_crud_functions():
    """Test to see what CRUD functions are actually available"""
    try:
        # Import the crud module dynamically
        crud_module = importlib.import_module('db.crud')
        
        # Get all functions that don't start with underscore
        available_functions = []
        for name in dir(crud_module):
            if not name.startswith('_'):
                obj = getattr(crud_module, name)
                if callable(obj):
                    available_functions.append(name)
        
        print(f"\n=== AVAILABLE CRUD FUNCTIONS ===")
        for func in sorted(available_functions):
            func_obj = getattr(crud_module, func)
            if hasattr(func_obj, '__annotations__') or inspect.isfunction(func_obj):
                try:
                    sig = inspect.signature(func_obj)
                    print(f"✅ {func}{sig}")
                except:
                    print(f"✅ {func}(...)")
        
        print(f"\n=== TOTAL FUNCTIONS FOUND: {len(available_functions)} ===")
        
    except ImportError as e:
        print(f"❌ Error importing db.crud: {e}")
        
    # Check models too
    try:
        models_module = importlib.import_module('db.models')
        
        model_classes = []
        for name in dir(models_module):
            if not name.startswith('_'):
                obj = getattr(models_module, name)
                if inspect.isclass(obj) and hasattr(obj, '__tablename__'):
                    model_classes.append(name)
        
        print(f"\n=== AVAILABLE MODEL CLASSES ===")
        for model in sorted(model_classes):
            print(f"📊 {model}")
            
        print(f"\n=== TOTAL MODELS FOUND: {len(model_classes)} ===")
            
    except ImportError as e:
        print(f"❌ Error importing db.models: {e}")
    
    # This test always passes - it's just for inspection
    assert True

def test_check_database_models():
    """Check what models are available and their fields"""
    try:
        from db.models import Base
        
        print(f"\n=== DATABASE MODELS INSPECTION ===")
        
        # Get all model classes
        for name, obj in Base.registry._class_registry.items():
            if hasattr(obj, '__tablename__'):
                print(f"\n📊 Model: {name}")
                print(f"   Table: {obj.__tablename__}")
                
                # Get columns
                if hasattr(obj, '__table__'):
                    print("   Columns:")
                    for column in obj.__table__.columns:
                        print(f"     - {column.name}: {column.type}")
                        
    except Exception as e:
        print(f"❌ Error inspecting models: {e}")
    
    assert True

if __name__ == "__main__":
    # Run the inspection
    test_check_existing_crud_functions()
    test_check_database_models()