import torch
import gymnasium as gym
from Model import HardwareDQN  
import os

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME    = "CartPole-v1"
CHECKPOINT  = "<MODEL-PATH>"   # Load the saved best model
N_EVAL      = 50

env = gym.make(ENV_NAME, render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = HardwareDQN(state_dim, n_actions).to(DEVICE)
assert os.path.isfile(CHECKPOINT), f"Checkpoint not found: {CHECKPOINT}"
policy_net.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
policy_net.eval()

for ep in range(1, N_EVAL+1):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            action = policy_net(state).argmax(dim=1).item()
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        done = term or trunc
        state = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    print(f"Eval episode {ep:2d}: length = {total_reward}")
