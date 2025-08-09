# in api/routes/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api import schemas
from db import crud
from db.session import get_db

router = APIRouter()

@router.post("/users/", response_model=schemas.User, tags=["Users"])
def create_new_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)