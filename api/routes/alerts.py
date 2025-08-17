from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from api import schemas
from db import crud, models
from db.session import get_db
from api.routes.auth import get_current_user
from services.alert_service import AlertService

router = APIRouter()

# Initialize alert service
alert_service = AlertService()

@router.post("/alerts/", response_model=schemas.Alert, tags=["Alerts"])
def create_alert(
    alert: schemas.AlertCreate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Create a new portfolio alert"""
    try:
        # Verify portfolio ownership if portfolio_id provided
        if alert.portfolio_id:
            portfolio = db.query(models.Portfolio).filter(
                models.Portfolio.id == alert.portfolio_id,
                models.Portfolio.user_id == current_user.id
            ).first()
            if not portfolio:
                raise HTTPException(status_code=404, detail="Portfolio not found")
        
        # Create alert
        db_alert = models.Alert(
            user_id=current_user.id,
            **alert.dict()
        )
        
        db.add(db_alert)
        db.commit()
        db.refresh(db_alert)
        
        return db_alert
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")


@router.get("/alerts/", response_model=List[schemas.Alert], tags=["Alerts"])
def get_user_alerts(
    portfolio_id: Optional[int] = Query(None, description="Filter by portfolio ID"),
    alert_type: Optional[schemas.AlertTypeEnum] = Query(None, description="Filter by alert type"),
    status: Optional[schemas.AlertStatusEnum] = Query(None, description="Filter by status"),
    active_only: bool = Query(True, description="Show only active alerts"),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Get user's alerts with optional filtering"""
    try:
        query = db.query(models.Alert).filter(models.Alert.user_id == current_user.id)
        
        if portfolio_id:
            query = query.filter(models.Alert.portfolio_id == portfolio_id)
        
        if alert_type:
            query = query.filter(models.Alert.alert_type == alert_type)
        
        if status:
            query = query.filter(models.Alert.status == status)
        
        if active_only:
            query = query.filter(models.Alert.is_active == True)
        
        alerts = query.order_by(models.Alert.created_at.desc()).all()
        return alerts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch alerts: {str(e)}")


@router.get("/alerts/{alert_id}", response_model=schemas.Alert, tags=["Alerts"])
def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Get specific alert by ID"""
    alert = db.query(models.Alert).filter(
        models.Alert.id == alert_id,
        models.Alert.user_id == current_user.id
    ).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return alert


@router.put("/alerts/{alert_id}", response_model=schemas.Alert, tags=["Alerts"])
def update_alert(
    alert_id: int,
    alert_update: schemas.AlertUpdate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Update an existing alert"""
    try:
        alert = db.query(models.Alert).filter(
            models.Alert.id == alert_id,
            models.Alert.user_id == current_user.id
        ).first()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Update only provided fields
        update_data = alert_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(alert, field, value)
        
        db.commit()
        db.refresh(alert)
        
        return alert
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update alert: {str(e)}")


@router.delete("/alerts/{alert_id}", tags=["Alerts"])
def delete_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Delete an alert"""
    try:
        alert = db.query(models.Alert).filter(
            models.Alert.id == alert_id,
            models.Alert.user_id == current_user.id
        ).first()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        db.delete(alert)
        db.commit()
        
        return {"message": "Alert deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete alert: {str(e)}")


@router.post("/alerts/{alert_id}/pause", response_model=schemas.Alert, tags=["Alerts"])
def pause_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Pause an active alert"""
    try:
        alert = db.query(models.Alert).filter(
            models.Alert.id == alert_id,
            models.Alert.user_id == current_user.id
        ).first()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.status = models.AlertStatus.PAUSED
        alert.is_active = False
        
        db.commit()
        db.refresh(alert)
        
        return alert
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to pause alert: {str(e)}")


@router.post("/alerts/{alert_id}/resume", response_model=schemas.Alert, tags=["Alerts"])
def resume_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Resume a paused alert"""
    try:
        alert = db.query(models.Alert).filter(
            models.Alert.id == alert_id,
            models.Alert.user_id == current_user.id
        ).first()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.status = models.AlertStatus.ACTIVE
        alert.is_active = True
        
        db.commit()
        db.refresh(alert)
        
        return alert
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to resume alert: {str(e)}")


@router.post("/alerts/check", tags=["Alerts"])
def check_alerts_manually(
    portfolio_id: Optional[int] = Query(None, description="Check alerts for specific portfolio"),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(get_current_user)
):
    """Manually trigger alert checking (for testing/debugging)"""
    try:
        triggered_alerts = alert_service.check_user_alerts(db, current_user.id, portfolio_id)
        
        return {
            "message": f"Alert check completed. {len(triggered_alerts)} alerts triggered.",
            "triggered_alerts": [alert.id for alert in triggered_alerts]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert check failed: {str(e)}")