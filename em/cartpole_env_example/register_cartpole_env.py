# # https://isaac-sim.github.io/IsaacLab/source/tutorials/03_envs/register_rl_env_gym.html

# import gymnasium as gym

# from . import agents
# from .cartpole_camera_env_cfg import CartpoleDepthCameraEnvCfg, CartpoleRGBCameraEnvCfg
# from .cartpole_env_cfg import CartpoleEnvCfg

# ##
# # Register Gym environments.
# ##

# gym.register(
#     id="Isaac-Cartpole-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": CartpoleEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Cartpole-RGB-Camera-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": CartpoleRGBCameraEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Cartpole-Depth-Camera-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": CartpoleDepthCameraEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
#     },
# )