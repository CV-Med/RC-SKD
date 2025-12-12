import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from cut_Res import target_model
# from cut_Dens import target_model


# 定义自定义环境
class MaximizeOutputEnv(gym.Env):
    def __init__(self, target_model):
        super(MaximizeOutputEnv, self).__init__()
        self.target_model = target_model

        # 状态空间输入值，每个值在[0.3, 0.5]范围内
        self.observation_space = spaces.Box(low = 0.3, high = 0.5, shape = (4,), dtype = np.float32)

        # 动作空间连续动作（每个数值在[-0.1, 0.1]之间的小调整）
        self.action_space = spaces.Box(low = -0.1, high = 0.1, shape = (4,), dtype = np.float32)

        # 初始化输入张量
        self.state = 0.3 + (0.5 - 0.3) * np.random.rand(4)
        self.state = self.state.astype(np.float32)  # 转换为 float32 类型

    def reset(self, *args, **kwargs):
        # 重置环境状态
        self.state = 0.3 + (0.5 - 0.3) * np.random.rand(4)
        self.state = self.state.astype(np.float32)  # 转换为 float32 类型
        return self.state, {}  # 返回状态和空字典

    def step(self, action):
        # 根据动作调整输入张量，并限制范围在[0, 1]
        self.state = np.clip(self.state + action, 0.3, 0.5)

        # 计算目标模型的输出作为奖励
        reward = float(self.target_model(self.state) * 1000)

        # 假设没有终止条件
        done = False
        truncated = False
        return self.state, reward, done, truncated, {}


if __name__ == '__main__':
    # 训练环境测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")

    env = make_vec_env(lambda: MaximizeOutputEnv(target_model), n_envs = 1)

    # 使用PPO训练
    # model = PPO("MlpPolicy", env, verbose = 1)

    # 定义DDPG智能体的动作噪声
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))

    # 初始化DDPG模型
    model = DDPG("MlpPolicy", env, action_noise = action_noise, verbose = 1)
    model.learn(total_timesteps = 100)
    count = 0

    # 训练完成后保存模型
    model_save_path = "DDPG_Brain_3_R34.zip"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # # 从保存的文件中加载模型
    # loaded_model_path = "trained_ppo_model.zip"
    # model = DDPG.load(loaded_model_path)
    # print("Model loaded successfully")
    # 测试训练后的模型
    obs = env.reset()
    for i in range(20):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)  # 解包四个值
        print(f"Step {i + 1} -> 动作: {action}, 下一个状态: {obs}, 奖励: {reward}")