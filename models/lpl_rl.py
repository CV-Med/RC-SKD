import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from models.cut_res import target_model as target_model_res
from models.cut_dens import target_model as target_model_dens


class PruningEnv(gym.Env):
    def __init__(self, target_model_func, lam=1000, arch=50,
                 num_classes=100, dataset='CIFAR100', save_path=None):
        super().__init__()
        self.target_model_func = target_model_func
        self.lam = lam
        self.arch = arch
        self.num_classes = num_classes
        self.dataset = dataset
        self.save_path = save_path
        self.n_actions = 4
        self.observation_space = spaces.Box(low=0.3, high=0.5, shape=(self.n_actions,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(self.n_actions,), dtype=np.float32)
        self.state = np.random.uniform(0.3, 0.5, self.n_actions).astype(np.float32)

    def reset(self, *args, **kwargs):
        self.state = np.random.uniform(0.3, 0.5, self.n_actions).astype(np.float32)
        return self.state, {}

    def step(self, action):
        self.state = np.clip(self.state + action, 0.3, 0.5)
        acc = float(self.target_model_func(
            self.state, arch=self.arch,
            num_classes=self.num_classes, dataset=self.dataset,
            save_path=self.save_path
        ))
        mean_pruning = float(np.mean(self.state))
        reward = self.lam * acc * mean_pruning
        done = False
        truncated = False
        return self.state, reward, done, truncated, {}


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use device is {device}")

    import os
    os.makedirs('./weights', exist_ok=True)
    save_path = './weights/pruned_CIFAR100_resnet50.pth'

    env = make_vec_env(
        lambda: PruningEnv(target_model_res, lam=1000, arch=50,
                          num_classes=100, dataset='CIFAR100',
                          save_path=save_path),
        n_envs=1
    )
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=1000)
    model_save_path = "DDPG_CIFAR100_resnet50.zip"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    obs = env.reset()
    for i in range(20):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f"Step {i + 1} -> action: {action}, next state: {obs}, reward: {reward}")
