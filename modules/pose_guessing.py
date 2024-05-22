from abc import ABC, abstractmethod
import numpy as np

class PoseGuesserBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, last_poses):
        return np.identity(4)

class PoseGuesserLastPose(PoseGuesserBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, last_poses):
        if len(last_poses) < 1:
            return np.identity(4)
        
        return np.copy(last_poses[-1])

class PoseGuesserConstantVelocity(PoseGuesserBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, last_poses):
        if len(last_poses) < 2:
            return np.identity(4)

        pose_0 = last_poses[-2]
        pose_1 = last_poses[-1]

        try:
            M = pose_1 @ np.linalg.inv(pose_0)
        except:
            breakpoint()
            
        return M @ pose_1