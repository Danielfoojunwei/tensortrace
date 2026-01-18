import pytest
import numpy as np
from tensorguard import create_client, Demonstration
from tensorguard.utils.exceptions import ValidationError

def test_edge_client_init():
    client = create_client(model_type="pi0")
    assert client.config.model_type == "pi0"
    assert client.get_status().total_submissions == 0

def test_client_missing_adapter():
    client = create_client()
    demo = Demonstration(observations=[np.zeros(10)], actions=[np.zeros(2)])
    client.add_demonstration(demo)
    
    with pytest.raises(ValidationError):
        client.process_round()

def test_client_with_adapter():
    from tensorguard.core.adapters import MoEAdapter
    client = create_client()
    client.set_adapter(MoEAdapter())
    
    client.add_demonstration(Demonstration(observations=[np.zeros(10)], actions=[np.zeros(2)]))
    encrypted = client.process_round()
    
    assert encrypted is not None
    assert isinstance(encrypted, bytes)
    assert client.get_status().total_submissions == 1

