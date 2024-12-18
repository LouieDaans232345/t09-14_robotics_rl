from stable_baselines3.common.env_checker import check_env
from t10envw import OT2Env

wrapped_env = OT2Env()
check_env(wrapped_env)
