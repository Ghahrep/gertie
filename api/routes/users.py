# in api/routes/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api import schemas
from db import crud
from db.session import get_db
from api.routes.auth import get_current_user
from typing import Dict, Any

router = APIRouter()

@router.post("/users/", response_model=schemas.User, tags=["Users"])
def create_new_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)


@router.get("/users/preferences", response_model=schemas.UserPreferences, tags=["User Preferences"])
def get_user_preferences(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Get current user's preferences"""
    try:
        user = crud.get_user_by_email(db, email=current_user.email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Return preferences or defaults
        preferences = user.preferences or {}
        
        # Ensure all sections exist with defaults
        default_prefs = {
            "display": {
                "currency": "USD",
                "date_format": "MM/DD/YYYY",
                "theme": "dark", 
                "timezone": "America/New_York"
            },
            "notifications": {
                "email_alerts": True,
                "price_threshold_pct": 5.0,
                "risk_alerts": True,
                "ai_insights": True,
                "daily_summary": False
            },
            "dashboard": {
                "refresh_interval": 60,
                "default_page": "overview",
                "charts_animation": True,
                "show_benchmarks": True
            },
            "privacy": {
                "data_sharing": False,
                "analytics_tracking": True,
                "performance_monitoring": True
            }
        }
        
        # Merge user preferences with defaults
        for section, defaults in default_prefs.items():
            if section not in preferences:
                preferences[section] = defaults
            else:
                # Fill in missing keys
                for key, default_value in defaults.items():
                    if key not in preferences[section]:
                        preferences[section][key] = default_value
        
        return preferences
        
    except Exception as e:
        print(f"Error fetching user preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch preferences")


@router.put("/users/preferences", response_model=schemas.UserPreferences, tags=["User Preferences"])
def update_user_preferences(
    preferences_update: schemas.UserPreferencesUpdate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Update user's preferences (partial updates supported)"""
    try:
        user = crud.get_user_by_email(db, email=current_user.email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get current preferences or initialize
        current_prefs = user.preferences or {}
        
        # Update only the sections that were provided
        update_data = preferences_update.dict(exclude_unset=True)
        
        for section, section_data in update_data.items():
            if section not in current_prefs:
                current_prefs[section] = {}
            current_prefs[section].update(section_data)
        
        # Save updated preferences
        user.preferences = current_prefs
        db.commit()
        db.refresh(user)
        
        return current_prefs
        
    except Exception as e:
        print(f"Error updating user preferences: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.post("/users/preferences/reset", response_model=schemas.UserPreferences, tags=["User Preferences"])
def reset_user_preferences(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Reset user preferences to defaults"""
    try:
        user = crud.get_user_by_email(db, email=current_user.email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Reset to defaults
        default_preferences = {
            "display": {
                "currency": "USD",
                "date_format": "MM/DD/YYYY",
                "theme": "dark",
                "timezone": "America/New_York"
            },
            "notifications": {
                "email_alerts": True,
                "price_threshold_pct": 5.0,
                "risk_alerts": True,
                "ai_insights": True,
                "daily_summary": False
            },
            "dashboard": {
                "refresh_interval": 60,
                "default_page": "overview",
                "charts_animation": True,
                "show_benchmarks": True
            },
            "privacy": {
                "data_sharing": False,
                "analytics_tracking": True,
                "performance_monitoring": True
            }
        }
        
        user.preferences = default_preferences
        db.commit()
        db.refresh(user)
        
        return default_preferences
        
    except Exception as e:
        print(f"Error resetting user preferences: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to reset preferences")