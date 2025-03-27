from .cartpole_continuous_env import ContinuousCartPoleEnv

def create_continuous_cartpole_env():
    return ContinuousCartPoleEnv()

__all__ = [
    'create_continuous_cartpole_env',
]

