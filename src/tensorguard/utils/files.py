import os
import shutil
import tempfile
from pathlib import Path
from typing import Union, Optional

from .logging import get_logger

logger = get_logger(__name__)

def atomic_write(target_path: Union[str, Path], data: Union[str, bytes]):
    """
    Writes data to a file atomically via a temporary file.
    Prevents file corruption if the process is interrupted.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Use the same directory as the target to ensure same-device os.replace
    with tempfile.NamedTemporaryFile(
        dir=target.parent, 
        delete=False, 
        mode='w' if isinstance(data, str) else 'wb',
        suffix=".tmp"
    ) as tf:
        tf.write(data)
        temp_name = tf.name
        
    try:
        os.replace(temp_name, target)
    except Exception as e:
        logger.error(f"Failed to perform atomic write to {target}: {e}")
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise

def sanitize_path(path_str: str, base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Sanitizes a path to prevent directory traversal attacks.
    If base_dir is provided, ensures the path is relative to base_dir.
    """
    # Remove any platform-specific traversal characters
    safe_name = os.path.basename(path_str)
    
    if base_dir:
        return Path(base_dir) / safe_name
    return Path(safe_name)
