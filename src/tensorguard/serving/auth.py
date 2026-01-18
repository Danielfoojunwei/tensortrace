"""
Authentication utilities for the MOAI Serving Gateway.
"""
from fastapi import Depends
from ..platform.auth import get_current_user
from ..platform.models.core import User

async def get_current_tenant(user: User = Depends(get_current_user)) -> str:
    """
    Extract tenant ID from the authenticated user.
    """
    return user.tenant_id
