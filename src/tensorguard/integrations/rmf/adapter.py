"""
Open-RMF Adapter for TensorGuard MOAI
"""

import time
import requests
import json
from dataclasses import dataclass
from typing import Dict, Any

from ...utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class RmfTask:
    task_id: str
    robot_id: str
    type: str
    payload: Dict[str, Any]

class RmfAdapter:
    """
    Simulates binding between Open-RMF fleet adapter and MOAI Inference Service.
    """
    
    def __init__(self, moai_gateway_url: str, tenant_token: str):
        self.gateway_url = moai_gateway_url
        self.token = tenant_token
        
    def process_task_request(self, task: RmfTask) -> bool:
        """
        Receive a task from RMF, route to MOAI to determine policy parameters.
        """
        logger.info(f"RMF Task Received: {task.task_id} type={task.type}")
        
        # 1. Transform Task -> Tensor Input (Mock)
        # In reality, this might involve fetching the robot's camera feed or state
        # Here we mock an encrypted payload
        mock_ciphertext = "bW9ja19jaXBoZXJ0ZXh0" # "mock_ciphertext" base64
        mock_key = "bW9ja19rZXk="
        
        # 2. Call MOAI Inference
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {
                "ciphertext_base64": mock_ciphertext,
                "eval_keys_base64": mock_key,
                "metadata": {"task_id": task.task_id}
            }
            
            resp = requests.post(
                f"{self.gateway_url}/v1/infer", 
                json=payload, 
                headers=headers,
                timeout=5.0
            )
            resp.raise_for_status()
            
            result = resp.json()
            logger.info(f"MOAI Inference Success: {result['compute_time_ms']}ms")
            
            # 3. Apply Policy (Mock)
            # Robot would decrypt result['result_ciphertext_base64'] and act
            
            return True
        except Exception as e:
            logger.error(f"Failed to process RMF task via MOAI: {e}")
            return False
