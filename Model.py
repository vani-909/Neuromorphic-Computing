import math
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import matplotlib
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from moviepy.editor import ImageSequenceClip

# Actual hardware metrics extracted from measurements
ON_OFF_RATIO = 1.57
WRITE_NOISE_STD = 5.16e-07
READ_NOISE_STD = 4.31e-08
LINEARITY = 1.0
SYMMETRY = True


# Fitted polynomial models of conductance change (slope of drain current over time)
# For example:
# "10.9"   → pulse, +0.9V
# "1-0.45" → pulse, −0.45V
# "00.9"   → delay, +0.9V
# "0-0.3"  → delay, −0.3V

results = {'10.9': [np.float64(0.00011886583678895393), np.array([ 2.65028370e+02, -1.12005284e-01,  9.56899283e-06]), np.array([[ 3.41081904e+02, -1.40178683e-01,  1.41943360e-05],
       [-1.40178683e-01,  5.77947051e-05, -5.87107555e-09],
       [ 1.41943360e-05, -5.87107555e-09,  5.98452594e-13]])], '10.45': [np.float64(0.00016345491931659973), np.array([ 1.47735029e+02, -7.74527358e-02,  8.71291847e-06]), np.array([[ 7.72586669e+01, -3.17493224e-02,  3.23216994e-06],
       [-3.17493224e-02,  1.30689124e-05, -1.33258731e-09],
       [ 3.23216994e-06, -1.33258731e-09,  1.36097623e-13]])], '10.3': [np.float64(0.00018394020332966517), np.array([ 4.80155874e+01, -4.05292413e-02,  5.83039758e-06]), np.array([[ 1.11516367e+02, -4.56996700e-02,  4.66421120e-06],
       [-4.56996700e-02,  1.87407032e-05, -1.91397779e-09],
       [ 4.66421120e-06, -1.91397779e-09,  1.95598388e-13]])], '1-0.9': [np.float64(0.00026003875557833057), np.array([ 1.15944204e+02, -1.15700020e-01,  2.22463244e-05]), np.array([[ 1.91178939e+02, -8.12585681e-02,  8.49919839e-06],
       [-8.12585681e-02,  3.46429721e-05, -3.63588463e-09],
       [ 8.49919839e-06, -3.63588463e-09,  3.83126191e-13]])], '1-0.45': [np.float64(0.0002492345648416061), np.array([ 1.22287678e+02, -9.21124970e-02,  1.53613682e-05]), np.array([[ 7.97182020e+01, -3.42316063e-02,  3.63838503e-06],
       [-3.42316063e-02,  1.47264642e-05, -1.56844986e-09],
       [ 3.63838503e-06, -1.56844986e-09,  1.67438006e-13]])], '1-0.3': [np.float64(0.00024176044228483494), np.array([ 1.62371063e+02, -1.04453074e-01,  1.57623395e-05]), np.array([[ 2.05092633e+02, -8.93237071e-02,  9.66763423e-06],
       [-8.93237071e-02,  3.89452088e-05, -4.22010123e-09],
       [ 9.66763423e-06, -4.22010123e-09,  4.57891652e-13]])], '00.9': [np.float64(0.0002555617843892662), np.array([-7.78162567e+00,  1.99715409e-03, -2.16408543e-09]), np.array([[ 6.50773033e+01, -2.64767143e-02,  2.65389768e-06],
       [-2.64767143e-02,  1.08069748e-05, -1.08677951e-09],
       [ 2.65389768e-06, -1.08677951e-09,  1.09668314e-13]])], '00.45': [np.float64(0.00023529982087532397), np.array([-3.55743010e+01,  1.40563603e-02, -1.33785212e-06]), np.array([[ 8.13358446e+01, -3.31383239e-02,  3.34628985e-06],
       [-3.31383239e-02,  1.35227011e-05, -1.36760309e-09],
       [ 3.34628985e-06, -1.36760309e-09,  1.38522597e-13]])], '00.3': [np.float64(0.00022605881252930259), np.array([-5.15965675e+01,  2.10562616e-02, -2.12323543e-06]), np.array([[ 5.09530141e+02, -2.07419577e-01,  2.10381076e-05],
       [-2.07419577e-01,  8.44879749e-05, -8.57445489e-09],
       [ 2.10381076e-05, -8.57445489e-09,  8.70693674e-13]])], '0-0.9': [np.float64(0.00026470970595861255), np.array([-2.05803167e+01,  9.73904812e-03, -1.13593249e-06]), np.array([[ 1.86982629e+02, -8.06491110e-02,  8.58040479e-06],
       [-8.06491110e-02,  3.48748597e-05, -3.72104187e-09],
       [ 8.58040479e-06, -3.72104187e-09,  3.98334608e-13]])], '0-0.45': [np.float64(0.00030845311348857936), np.array([-1.23612393e+01,  6.44789895e-03, -8.12785123e-07]), np.array([[ 2.07161987e+02, -9.01196298e-02,  9.72153183e-06],
       [-9.01196298e-02,  3.92629500e-05, -4.24244813e-09],
       [ 9.72153183e-06, -4.24244813e-09,  4.59254225e-13]])], '0-0.3': [np.float64(0.00024537588997422825), np.array([-4.98502333e+01,  2.22827221e-02, -2.46619375e-06]), np.array([[ 1.27760377e+03, -5.62721409e-01,  6.16733933e-05],
       [-5.62721409e-01,  2.48062197e-04, -2.72122967e-08],
       [ 6.16733933e-05, -2.72122967e-08,  2.98818147e-12]])]}


# Play around with these 2 values obtained during measurement 
OPERATING_VOLTAGE = 0.9
PULSE = 1

minimum, popt, pcov = results[str(PULSE)+str(OPERATING_VOLTAGE)]
maximum, popt, pcov = results[str(PULSE)+str(-1*OPERATING_VOLTAGE)]
width = (maximum - minimum) / 2

# Convert a conductance value x into a noisy slope (ΔI/Δt) using the quadratic fit  a·x² + b·x + c  and Gaussian device noise.
def rand_conversion(x, popt, std):
    slope = popt[0]*x**2 + popt[1]*x + popt[2]
    return np.random.normal(loc=slope, scale=std)
   

# Adam with fitted-polynomial hardware conversion.
class HardwareAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, 
                 pulse=PULSE, v_amp=OPERATING_VOLTAGE, max_gain=1.0, noise_clip=1e6):
        defaults = dict(lr=lr, betas=betas, eps=eps, pulse=pulse, v_amp=v_amp, max_gain=max_gain, noise_clip=noise_clip)
        super().__init__(params, defaults)

    # Convert an Adam step tensor to hardware-scale gain tensor {step == gradient}
    @staticmethod
    def _poly_gain(w, step, *, pulse, v_amp, max_gain, noise_clip):
        # Vectorise all tensors for GPU.
        w_flat   = w.reshape(-1)
        g_flat   = step.reshape(-1)
        upd      = torch.zeros_like(w_flat)

        # masks for + and – polarity
        pos_mask = g_flat > 0
        neg_mask = g_flat < 0
        if not (pos_mask.any() or neg_mask.any()):
            return upd.reshape_as(w)           # nothing to update if grad=0

        def _gain(mask, key):
            if not mask.any() or key not in results:
                return torch.zeros(mask.sum(), device=w.device)

            # Fitted polynomial coefficients (a, b, c) and cov.
            _, popt_np, pcov_np = results[key]
            popt = torch.tensor(popt_np, dtype=torch.float32, device=w.device)
            pcov = torch.tensor(pcov_np, dtype=torch.float32, device=w.device)

            w_sub = w_flat[mask]

            # Prepare the non-linear, weight-dependent noise model
            scale   = torch.stack((w_sub**2, w_sub, torch.ones_like(w_sub)), 1)
            pcov_sc = pcov.unsqueeze(0) * scale.unsqueeze(2) * scale.unsqueeze(1)
            sigma   = torch.sqrt(pcov_sc.sum((1, 2))).clamp(max=noise_clip)

            # Evaluate the polynomial slope at each weight and add noise
            slope   = popt[0]*w_sub**2 + popt[1]*w_sub + popt[2]
            slope  += torch.normal(0.0, sigma)

            # Normalization
            width = results[f"{pulse}{-v_amp}"][0] - results[f"{pulse}{ v_amp}"][0]            
            return (slope / width).abs().clamp(max=max_gain)

        gain_pos = _gain(pos_mask, f"{pulse}{-v_amp}")   # –V pulse
        gain_neg = _gain(neg_mask, f"{pulse}{ v_amp}")   # +V pulse

        if pos_mask.any():
            upd[pos_mask] = gain_pos * g_flat[pos_mask]
        if neg_mask.any():
            upd[neg_mask] = gain_neg * g_flat[neg_mask]

        upd[~torch.isfinite(upd) | (upd.abs() > noise_clip)] = 0.0    # Sanity Check
        return upd.reshape_as(w)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr, betas, eps = group['lr'], group['betas'], group['eps']
            pulse     = group['pulse']
            v_amp     = group['v_amp']
            max_gain  = group['max_gain']
            noise_clip = group['noise_clip']

            beta1, beta2 = betas

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.detach()

                state = self.state[p]

                # state initialisation 
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Adam update 
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                bias_corr1 = 1 - beta1 ** t
                bias_corr2 = 1 - beta2 ** t

                step_tensor = (exp_avg / bias_corr1) / (
                    exp_avg_sq.sqrt() / math.sqrt(bias_corr2) + eps)

                # Convert Adam step -> hardware Δw
                delta_w = self._poly_gain(p.data, step_tensor, pulse=pulse, v_amp=v_amp, max_gain=max_gain, noise_clip=noise_clip)

                # Parameter update
                p.data.add_(delta_w, alpha=-lr)

        return loss



class LinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, on_off_ratio, write_noise_std, read_noise_std, linearity, symmetry):
        super().__init__(in_features, out_features)
        self.on_off_ratio = on_off_ratio
        self.write_noise_std = write_noise_std
        self.read_noise_std = read_noise_std
        self.linearity = linearity
        self.symmetry = symmetry

        # physical conductance limits (all positive)
        self.Gmin = 1.0 / on_off_ratio
        self.Gmax = 1.0

        # Cart-Pole needs signed weights to learn properly (logical weights)
        self.weight_range = (-float('inf'), float('inf'))


    def _logic2conductance(self, w_logic: torch.Tensor):
        """
        Map signed logical weight → two positive conductances.
        G⁺ =  0.5·(|w|+w)·ΔG + Gmin
        G⁻ =  0.5·(|w|-w)·ΔG + Gmin
        so that  G⁺-G⁻ = w·ΔG   with  ΔG = Gmax-Gmin
        """
        deltaG = self.Gmax - self.Gmin
        G_pos  = 0.5 * (w_logic.abs() +  w_logic) * deltaG + self.Gmin
        G_neg  = 0.5 * (w_logic.abs() -  w_logic) * deltaG + self.Gmin
        return G_pos, G_neg



    def forward(self, input):
        Gp, Gn = self._logic2conductance(self.weight)           

        if self.linearity != 1.0:               # Accounting for the non-linearity in the curve (pulse -> Conductance)
            Gp = torch.pow(Gp, self.linearity)
            Gn = torch.pow(Gn, self.linearity)

        deltaG = self.Gmax - self.Gmin
        # signed effective weight seen by the MAC
        w_eff = (Gp - Gn)/ deltaG

        if not self.symmetry:                  # Accounting for asymmetry (Different ΔG for the same +ve vs -ve pulse)
            mask = torch.rand_like(w_eff) < 0.5
            w_eff = w_eff * (1 - 0.1 * mask.float())   # Accounting for uncertainities like device variability, temperature, aging etc.

        noisy_w = w_eff + torch.randn_like(w_eff) * self.read_noise_std  # Applying read noise
        return F.linear(input, noisy_w, self.bias)
    


    def apply_write_noise(self):                
        with torch.no_grad():                   # Disable gradient tracking since it's hardware noise, not learning
            noise = torch.randn_like(self.weight) * self.write_noise_std
            self.weight += noise                # Applying write noise

            self.weight.clamp_(*self.weight_range)  # Ensures weights remain within physical device range [1/ON_OFF, 1]



# DQN with hardware-aware synapses
class HardwareDQN(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.fc1 = LinearLayer(n_obs, 128, ON_OFF_RATIO, WRITE_NOISE_STD, READ_NOISE_STD, LINEARITY, SYMMETRY)
        self.fc2 = LinearLayer(128, 128, ON_OFF_RATIO, WRITE_NOISE_STD, READ_NOISE_STD, LINEARITY, SYMMETRY)
        self.fc3 = LinearLayer(128, n_act, ON_OFF_RATIO, WRITE_NOISE_STD, READ_NOISE_STD, LINEARITY, SYMMETRY)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def apply_noise_and_clip(self):
        self.fc1.apply_write_noise()
        self.fc2.apply_write_noise()
        self.fc3.apply_write_noise()


# Training setup
is_ipython = 'inline' in matplotlib.get_backend()    # set up matplotlib
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(                              # if GPU is to be used
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
env = gym.make("CartPole-v1", render_mode="rgb_array")


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 128
GAMMA = 0.99      # Determines how much the agent values immediate vs. future rewards
EPS_START = 0.9
EPS_END = 0.05    # ε-greedy strategy
EPS_DECAY = 1000
TAU = 0.005      # Stability in Q-value targets (smaller, the better)
LR = 0.00025


best_score = -float('inf')  # for tracking best 100-episode average

n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

policy_net = HardwareDQN(n_observations, n_actions).to(device)
target_net = HardwareDQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)  -  SOFTWARE
optimizer = HardwareAdam(policy_net.parameters(), lr=2.5e-4, pulse=PULSE, v_amp=OPERATING_VOLTAGE, max_gain=5.0)

memory = ReplayMemory(10000)
steps_done = 0
episode_durations = []

def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)  # ε-greedy strategy
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def select_action_learned(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s, a) from policy network
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute target Q(s', a')
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values.detach()

    # Predict expected Q(s, a) using Bellman equation
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)

    # Compute Huber Loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()

    # Backpropagate the loss
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    
    optimizer.step()

    # Apply noise only every 20 steps for better stability
    global steps_done
    if steps_done % 20 == 0:
        policy_net.apply_noise_and_clip()


def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)


def record_video(env, filename, fps=15):
    frames = []
    done = False
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    img = env.render()
    frames.append(img)

    while not done:
        action = select_action_learned(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        img = env.render()
        frames.append(img)
        if not done:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename, fps=fps)
    

# Training loop
num_episodes = 300
for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        reward = torch.tensor([reward], device=device)
        next_state = None if done else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)

        if done:
            episode_durations.append(t + 1)
            plot_durations()

            # Save model if current 100-episode average is the best so far
            if len(episode_durations) >= 100:
                rolling_avg = sum(episode_durations[-100:]) / 100
                if rolling_avg > best_score:
                    best_score = rolling_avg
                    torch.save(policy_net.state_dict(), "best_model.pt")
                    print(f"New best model saved with 100-ep avg: {rolling_avg:.2f}")

            break


print("Training Complete")
policy_net.load_state_dict(torch.load("best_model.pt"))
print("Loaded best model for evaluation.")

plot_durations()

video_filename = f"Code/Videos/CartPole-v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
record_video(env, video_filename)

plt.ioff()
plt.show()