
import h5py
import numpy as np
import os
from typing import Dict, Any, List, Optional
from ..schemas.common import Demonstration

class FastUMIAdapter:
    """
    Data Adapter for the FastUMI-100K Dataset.
    Converts HDF5 robotic trajectories into TensorGuardFlow Demonstration objects.
    """
    def __init__(self, data_root: str):
        self.data_root = data_root
        
    def load_episode(self, episode_path: str) -> Demonstration:
        """Loads a single HDF5 episode into a Demonstration object."""
        with h5py.File(episode_path, 'r') as f:
            # Observational Data
            images = f['observations/images/front'][:] # (num_frames, 1920, 1080, 3)
            qpos = f['observations/qpos'][:] # (num_timesteps, 7)
            
            # Action Data
            action = f['action'][:] # (num_timesteps, 7)
            
            # Metadata / Attributes
            is_sim = f.attrs.get('sim', False)
            
            # Determine instruction based on path or attributes
            # In FastUMI, task folders usually denote the goal
            task_tag = os.path.basename(os.path.dirname(episode_path))
            instruction = f"Execute robotic task: {task_tag.replace('_', ' ')}"
            
            return Demonstration(
                observations=[{
                    "video_frame": images[i],
                    "qpos": qpos[i]
                } for i in range(len(qpos))],
                actions=action.tolist(),
                instruction=instruction,
                metadata={
                    "task": task_tag,
                    "sim": is_sim,
                    "provider": "FastUMI"
                }
            )

    def scan_dataset(self) -> List[str]:
        """Returns a list of all episode HDF5 paths in the data root."""
        paths = []
        for root, _, files in os.walk(self.data_root):
            for f in files:
                if f.endswith('.hdf5'):
                    paths.append(os.path.join(root, f))
        return paths

class FastUMISimulator:
    """Simulates the environment for a robot running FastUMI data."""
    def __init__(self, adapter: FastUMIAdapter):
        self.adapter = adapter
        self.episodes = self.adapter.scan_dataset()
        
    def get_random_demonstration(self, task_filter: str = None) -> Demonstration:
        """Returns a random demonstration, optionally filtered by task."""
        eligible = self.episodes
        if task_filter:
            eligible = [e for e in self.episodes if task_filter in e]
            
        if not eligible:
            # Fallback for demo (generate synthetic if no real files found)
            return self._generate_synthetic(task_filter or "general_task")
            
        path = np.random.choice(eligible)
        return self.adapter.load_episode(path)

    def _generate_synthetic(self, task: str) -> Demonstration:
        """Generates a synthetic demonstration matching FastUMI dims for demo purposes."""
        num_steps = 50
        return Demonstration(
            observations=[{
                "video_frame": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                "qpos": np.random.randn(7).tolist()
            } for _ in range(num_steps)],
            actions=np.random.randn(num_steps, 7).tolist(),
            instruction=f"Execute task: {task.replace('_', ' ')}",
            metadata={"task": task, "sim": True, "provider": "MockFastUMI"}
        )
