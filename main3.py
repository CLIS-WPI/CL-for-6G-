import os
import torch
import torch.nn as nn
import numpy as np
import random # Needed for PER sampling
from collections import deque, defaultdict # Added defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import tensorflow as tf
import time # To time runs
import csv # Added for saving raw data
from scipy import stats # Added for t-test

# --- Number of Runs --- Updated ---
NUM_RUNS = 15 # Set back to 10 runs, adjust if needed
BASE_SEED = 42 # Base seed for reproducibility across runs

# Ensure sionna is imported if not already done globally
try:
    import sionna as sn
except ImportError:
    print("Sionna library not found. Please install it.")
    exit()

# Ensure only GPU:0 is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Try to configure TensorFlow GPU settings
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]} (Memory growth enabled for TF)")
    else:
        print("No GPU found by TensorFlow.")
except Exception as e:
    print(f"Error configuring TensorFlow GPU: {e}")
    print("TensorFlow might not be using the GPU correctly.")

# Select device dynamically for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")


# --- Simulation Parameters --- Updated ---
NUM_UES = 5
NUM_ANTENNAS = 64
NUM_BEAMS = NUM_ANTENNAS
ROAD_LENGTH = 500
BS_POSITION = ROAD_LENGTH / 2
FREQ = 28e9
TX_POWER_DBM = 30
NOISE_POWER_DBM = -70
NUM_TIMESTEPS = 10000 # Kept from user's last code
EVAL_TIMESTEPS = 1000 # Kept from user's last code
TIMESTEP_DURATION = 0.01
BUFFER_SIZE = 50000 # Updated Capacity for PER
BATCH_SIZE = 64
LEARNING_RATE = 0.00005 # Reverted to potentially more stable value
GAMMA = 0.99
SNR_THRESHOLD = 14.0
TARGET_UPDATE_FREQ = 50 # Keep relatively frequent updates
PATH_LOSS_EXPONENT = 2.5
MAB_EXPLORATION_FACTOR = 2.0
VELOCITY_NORMALIZATION_FACTOR = 20.0
SNR_NORM_MIN = -10.0
SNR_NORM_MAX = 50.0
SMOOTHING_WINDOW = 50
GRU_HIDDEN_SIZE = 256 # Keep increased GRU hidden size
input_size = 5 # State dimension
# --- GRU Parameters --- Modified ---
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.2
# ---------------------------------

# --- PER Parameters --- Updated ---
PER_ALPHA = 0.6  # Reverted to value likely used in original paper
PER_BETA_START = 0.4 # Reverted to value likely used in original paper
PER_BETA_INCREMENT = (1.0 - PER_BETA_START) / NUM_TIMESTEPS # Updated based on new params
PER_EPSILON = 1e-5 # Small value added to priorities to ensure non-zero probability

# --- Epsilon-Greedy Parameters --- Added ---
EPSILON_START = 1.0
EPSILON_MIN = 0.01  # New minimum epsilon
EPSILON_DECAY = 0.9995 # Decay rate (reaches ~0.01 in ~9200 steps)

# --- MAB Epsilon --- Moved here for clarity ---
mab_epsilon = 1e-6 # Epsilon for MAB UCB1 calculation

# --- Time-Correlated Blockage Model Parameters ---
BLOCKAGE_ATTENUATION_DB = 25.0
P_BB = 0.85
P_UB = 0.03
# ---------------------------------------------

BANDWIDTH = 100e6

# --- Utility Function for Smoothing ---
def moving_average(data, window_size):
    if not isinstance(data, np.ndarray): data = np.array(data, dtype=np.float64)
    if data.size < window_size: return np.full(data.shape, np.nan)
    try:
        import pandas as pd
        s = pd.Series(data, dtype=float)
        return s.rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()
    except ImportError:
        kernel = np.ones(window_size) / float(window_size)
        smoothed = np.convolve(data, kernel, mode='same')
        half_window_floor = window_size // 2
        half_window_ceil = (window_size + 1) // 2
        smoothed[:half_window_floor] = np.nan
        end_idx_start = data.size - (half_window_ceil - 1)
        if end_idx_start < data.size :
            if end_idx_start >= 0: smoothed[end_idx_start:] = np.nan
        return smoothed
# ---------------------------------------

# Generate DFT-based codebook
def generate_codebook(num_antennas, num_beams):
    if num_beams != num_antennas: print(f"Warning: num_beams ({num_beams}) != num_antennas ({num_antennas}). Adjusting num_beams."); num_beams = num_antennas
    angles = np.linspace(-np.pi/2, np.pi/2, num_beams); codebook = np.zeros((num_antennas, num_beams), dtype=complex); antenna_indices = np.arange(num_antennas)
    for i, theta in enumerate(angles): steering_vector = np.exp(1j * np.pi * antenna_indices * np.sin(theta)); codebook[:, i] = steering_vector / np.sqrt(num_antennas)
    return codebook

CODEBOOK = generate_codebook(NUM_ANTENNAS, NUM_BEAMS)
BEAM_ANGLES = np.linspace(-np.pi/2, np.pi/2, NUM_BEAMS)

# --- SumTree for Prioritized Experience Replay ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity; self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity; self.write_ptr = 0; self.n_entries = 0
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2; self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left = 2 * idx + 1; right = left + 1
        if left >= len(self.tree): return idx
        if abs(self.tree[left]) < 1e-9 or s <= self.tree[left] + 1e-9:
             return self._retrieve(left, s)
        else:
             remaining_s = max(0.0, s - self.tree[left])
             return self._retrieve(right, remaining_s)
    def total(self): return self.tree[0]
    def add(self, priority, data):
        idx = self.write_ptr + self.capacity - 1; self.data[self.write_ptr] = data
        self.update(idx, priority); self.write_ptr += 1
        if self.write_ptr >= self.capacity: self.write_ptr = 0
        if self.n_entries < self.capacity: self.n_entries += 1
    def update(self, idx, priority):
        priority = max(priority, 0.0);
        change = priority - self.tree[idx]
        self.tree[idx] = priority; self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(0, s); data_idx = idx - self.capacity + 1
        if data_idx < 0 or data_idx >= self.capacity or self.data[data_idx] is None:
            valid_indices = [i for i, d in enumerate(self.data) if d is not None and i < self.n_entries]
            if not valid_indices: return (self.capacity - 1, 0.0, None)
            fallback_data_idx = random.choice(valid_indices)
            fallback_idx = fallback_data_idx + self.capacity - 1
            return (fallback_idx, self.tree[fallback_idx], self.data[fallback_data_idx])
        return (idx, self.tree[idx], self.data[data_idx])

# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity); self.alpha = alpha; self.epsilon = PER_EPSILON; self.max_priority = 1.0
    def add(self, state, action, reward, next_state):
        priority = self.max_priority ** self.alpha
        priority = max(priority, self.epsilon**self.alpha)
        self.tree.add(priority, (state, action, reward, next_state))
    def sample(self, batch_size, beta=0.4):
        batch = []; idxs = []; priorities = []; current_size = self.tree.n_entries
        if current_size < batch_size: return tuple(), tuple(), tuple(), tuple(), [], np.array([])
        total_priority = self.tree.total()
        if total_priority <= 0: return tuple(), tuple(), tuple(), tuple(), [], np.array([])
        num_sampled = 0; max_attempts = batch_size * 20
        segment = total_priority / batch_size
        sampled_indices_set = set()
        for i in range(batch_size):
            a = segment * i; b = segment * (i + 1); attempts = 0
            while attempts < max_attempts:
                attempts += 1; s = random.uniform(a, b)
                try: (idx, p, data) = self.tree.get(s)
                except Exception as e: print(f"Error during SumTree.get(s={s}): {e}"); continue
                if data is not None and idx not in sampled_indices_set:
                    priorities.append(p); batch.append(data); idxs.append(idx)
                    sampled_indices_set.add(idx); num_sampled += 1
                    break
        if num_sampled < batch_size:
            print(f"Warning: Could only sample {num_sampled}/{batch_size} unique valid entries.")
            if not batch: return tuple(), tuple(), tuple(), tuple(), [], np.array([])
        priorities = np.array(priorities)
        sampling_probabilities = priorities / total_priority if total_priority > 0 else np.full_like(priorities, 1.0 / len(priorities))
        is_weights = np.power(current_size * sampling_probabilities + 1e-8, -beta)
        min_possible_priority = np.min(self.tree.tree[self.tree.capacity - 1 : self.tree.capacity - 1 + current_size]) if current_size > 0 else self.epsilon**self.alpha
        max_weight = np.power(current_size * (min_possible_priority / total_priority if total_priority > 0 else 1.0) + 1e-8, -beta)
        is_weights /= max(max_weight, 1e-8)
        is_weights = np.nan_to_num(is_weights, nan=1.0)
        states, actions, rewards, next_states = zip(*batch)
        return states, actions, rewards, next_states, idxs, is_weights
    def update_priorities(self, batch_indices, td_errors):
        td_errors = np.asarray(td_errors);
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        priorities = np.maximum(priorities, self.epsilon**self.alpha)
        if len(priorities) > 0:
             current_max_error = np.max(np.abs(td_errors) + self.epsilon)
             self.max_priority = max(self.max_priority, current_max_error)
        for idx, priority_alpha in zip(batch_indices, priorities):
            if idx >= self.tree.capacity - 1 and idx < len(self.tree.tree):
                self.tree.update(idx, priority_alpha)
            else:
                print(f"Warning: Invalid index {idx} provided to update_priorities (Tree leaves range: {self.tree.capacity - 1} to {len(self.tree.tree) - 1}).")
    def __len__(self): return self.tree.n_entries

# --- GRU Q-Network Definition --- Modified ---
class GRUQNetwork(nn.Module):
    def __init__(self, input_size, gru_hidden_size, output_size, num_layers=GRU_NUM_LAYERS, dropout=GRU_DROPOUT): # Use defined params
        super(GRUQNetwork, self).__init__()
        self.input_size = input_size
        self.gru_hidden_size = gru_hidden_size
        self.output_size = output_size
        self.num_layers = num_layers # Store num_layers

        # Apply dropout only if num_layers > 1
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_size, gru_hidden_size, num_layers=num_layers, dropout=gru_dropout, batch_first=True)

        # FC layers remain the same
        self.fc1 = nn.Linear(gru_hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h_state=None):
        # Ensure input is 3D: (batch, seq_len, input_size)
        if x.dim() == 2: x = x.unsqueeze(1)
        elif x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)

        # Pass input and hidden state to GRU
        # h_state shape should be (num_layers, batch_size, hidden_size)
        gru_out, h_n = self.gru(x, h_state)

        # Use the output of the last time step from the last layer
        # gru_out shape: (batch, seq_len, hidden_size)
        last_step_out = gru_out[:, -1, :]

        # Pass through fully connected layers
        q = self.relu(self.fc1(last_step_out))
        q = self.relu(self.fc2(q))
        q = self.relu(self.fc3(q))
        q = self.fc4(q)
        # Return Q values and the final hidden state (h_n shape: num_layers, batch, hidden_size)
        return q, h_n
# --------------------------------------------

# --- Helper Functions (No changes needed below this point for epsilon/dropout) ---
def compute_path_loss(distances): min_distance = 1.0; distances = np.maximum(distances, min_distance); path_loss_db = 32.45 + 10 * PATH_LOSS_EXPONENT * np.log10(distances + 1e-9); path_loss_linear = 10 ** (-path_loss_db / 10); return path_loss_linear, path_loss_db
def compute_snr(h_channel_all_ues, beam_idx, ue_idx, extra_attenuation_db=0.0):
    if h_channel_all_ues is None or ue_idx >= h_channel_all_ues.shape[0] or beam_idx >= CODEBOOK.shape[1]: return -np.inf
    h_ue = h_channel_all_ues[ue_idx, :].astype(complex); beam = CODEBOOK[:, beam_idx].astype(complex);
    beam_norm = np.linalg.norm(beam)
    if abs(beam_norm - 1.0) > 1e-6: print(f"Warning: Beam {beam_idx} norm is {beam_norm}, should be 1.")
    if beam_norm > 1e-9 : beam = beam / beam_norm
    effective_signal_amplitude_sq = np.abs(np.vdot(h_ue, beam))**2
    noise_variance_relative = 10**((NOISE_POWER_DBM - TX_POWER_DBM) / 10)
    snr_linear = effective_signal_amplitude_sq / (noise_variance_relative + 1e-15)
    snr_db = 10 * np.log10(snr_linear + 1e-15) - extra_attenuation_db
    snr_db = np.clip(snr_db, -50.0, 50.0);
    return snr_db
def generate_channel(positions, channel_model_instance):
    if channel_model_instance is None: h = (np.random.randn(NUM_UES, NUM_ANTENNAS) + 1j * np.random.randn(NUM_UES, NUM_ANTENNAS)) / np.sqrt(2)
    else:
        try:
            h_tuple = channel_model_instance(batch_size=1, num_time_steps=1)
            h_sionna = h_tuple[0] if isinstance(h_tuple, (list, tuple)) else h_tuple
            h = h_sionna.numpy().reshape(NUM_UES, NUM_ANTENNAS)
        except Exception as e:
            print(f"Error during Sionna channel model execution: {e}. Falling back to Gaussian.")
            h = (np.random.randn(NUM_UES, NUM_ANTENNAS) + 1j * np.random.randn(NUM_UES, NUM_ANTENNAS)) / np.sqrt(2)
    distances = np.abs(positions - BS_POSITION); path_loss_linear, _ = compute_path_loss(distances);
    h_channel = h * np.sqrt(path_loss_linear[:, np.newaxis]);
    if not np.all(np.isfinite(h_channel)):
        print("Warning: Non-finite values detected in generated channel. Replacing with zeros.")
        h_channel = np.nan_to_num(h_channel, nan=0.0, posinf=0.0, neginf=0.0)
    return h_channel
def initial_beam_scan(h_channel):
    initial_snr = np.zeros(NUM_UES)
    if h_channel is None or not np.all(np.isfinite(h_channel)): return initial_snr
    for i in range(NUM_UES):
        best_snr = -float('inf')
        for beam_idx in range(NUM_BEAMS):
             snr = compute_snr(h_channel, beam_idx, i, extra_attenuation_db=0.0);
             if np.isfinite(snr) and snr > best_snr: best_snr = snr
        initial_snr[i] = best_snr if np.isfinite(best_snr) else -50.0
    return initial_snr
global_prev_positions = None
def update_positions(t):
    global global_prev_positions; positions = np.zeros(NUM_UES); velocities = np.zeros(NUM_UES)
    for i in range(NUM_UES):
        freq = 0.01 + i * 0.005
        movement_range = 200
        positions[i] = BS_POSITION + movement_range * np.sin(freq * t * TIMESTEP_DURATION)
        positions[i] = np.clip(positions[i], 0, ROAD_LENGTH)
    if global_prev_positions is not None: velocities = (positions - global_prev_positions) / TIMESTEP_DURATION
    else: velocities = np.zeros(NUM_UES)
    global_prev_positions = positions.copy(); return positions, velocities
def compute_relative_angles(positions): y_distance = 10.0; x_distance = positions - BS_POSITION; return np.arctan2(x_distance, y_distance)
def compute_distances(positions): distances_actual = np.abs(positions - BS_POSITION); normalized_distances = distances_actual / ROAD_LENGTH; return normalized_distances, distances_actual
def normalize_snr(snr_db): return np.clip((snr_db - SNR_NORM_MIN) / (SNR_NORM_MAX - SNR_NORM_MIN), 0.0, 1.0)
def normalize_velocity(velocity_mps): return np.clip(velocity_mps / VELOCITY_NORMALIZATION_FACTOR, -1.0, 1.0)
def compute_reward(throughput, snr, prev_snr, energy, accuracy_per_ue):
    w_tput = 1.0; w_stab = 0.2; w_energy = -0.3; w_acc = 0.5;
    r_norm = 1500.0; e_norm = 10.0
    throughput_reward = w_tput * (throughput / r_norm)
    stability_bonus = w_stab / (1 + abs(snr - prev_snr) + 1e-6)
    energy_penalty = w_energy * (max(0, energy) / e_norm)
    accuracy_reward = w_acc * np.mean(accuracy_per_ue)
    total_reward = throughput_reward + stability_bonus + energy_penalty + accuracy_reward
    if not np.isfinite(total_reward):
        print(f"Warning: Non-finite reward calculated. Tput: {throughput}, SNR: {snr}, PrevSNR: {prev_snr}, Energy: {energy}, Acc: {accuracy_per_ue}")
        return 0.0
    return total_reward
def compute_throughput(snr_db):
    if not np.isfinite(snr_db): snr_db = -50.0
    snr_linear = 10**(snr_db / 10.0); snr_linear = min(snr_linear, 1e15)
    throughput_bps = BANDWIDTH * np.log2(1 + snr_linear)
    return throughput_bps / 1e6
def compute_latency(avg_throughput_mbps):
    base_latency = 0.5; max_additional_latency = 5.0; threshold_throughput = 400.0; beta_latency = 0.01
    if not np.isfinite(avg_throughput_mbps): return base_latency + max_additional_latency
    else:
         exp_term = beta_latency * (avg_throughput_mbps - threshold_throughput); exp_term = np.clip(exp_term, -50, 50)
         latency = base_latency + max_additional_latency / (1 + np.exp(exp_term))
         return latency
def compute_energy(snr_db, distance_actual):
    base_energy_mj = 3.0; snr_factor_coeff = 0.1
    if not np.isfinite(snr_db): snr_db = -50.0
    snr_contribution = snr_factor_coeff * max(0, snr_db)
    energy = base_energy_mj + snr_contribution
    return max(0.1, energy)
def angle_heuristic_beam_switching(snr_heuristic, positions):
     _, distances_actual = compute_distances(positions)
     throughputs = np.array([compute_throughput(snr) for snr in snr_heuristic]); avg_throughput = np.nanmean(throughputs);
     latency = compute_latency(avg_throughput)
     energies = np.array([compute_energy(snr_heuristic[i], distances_actual[i]) for i in range(NUM_UES)]); avg_energy = np.nanmean(energies);
     accuracies = (snr_heuristic > SNR_THRESHOLD).astype(float); avg_accuracy = np.nanmean(accuracies);
     avg_snr = np.nanmean(snr_heuristic);
     return (latency if np.isfinite(latency) else np.nan, avg_throughput if np.isfinite(avg_throughput) else np.nan, avg_energy if np.isfinite(avg_energy) else np.nan, avg_accuracy if np.isfinite(avg_accuracy) else np.nan, avg_snr if np.isfinite(avg_snr) else np.nan)
def mab_ucb1_action(ue_idx, t, mab_counts, mab_values, exploration_factor, mab_epsilon_param):
    counts_ue = mab_counts[ue_idx, :]
    unexplored_arms = np.where(counts_ue == 0)[0]
    if len(unexplored_arms) > 0: return unexplored_arms[0]
    total_counts_ue = np.sum(counts_ue); log_total_counts = np.log(max(1, total_counts_ue))
    ucb_values = np.zeros(NUM_BEAMS)
    mean_rewards = mab_values[ue_idx, :] / (counts_ue + mab_epsilon_param)
    exploration_bonus = np.sqrt(exploration_factor * log_total_counts / (counts_ue + mab_epsilon_param))
    ucb_values = mean_rewards + exploration_bonus
    ucb_values = np.nan_to_num(ucb_values, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    best_arm = np.random.choice(np.where(ucb_values == np.max(ucb_values))[0])
    return best_arm
def mab_ucb1_beam_switching(snr_mab, positions):
     _, distances_actual = compute_distances(positions)
     throughputs = np.array([compute_throughput(snr) for snr in snr_mab]); avg_throughput = np.nanmean(throughputs);
     latency = compute_latency(avg_throughput)
     energies = np.array([compute_energy(snr_mab[i], distances_actual[i]) for i in range(NUM_UES)]); avg_energy = np.nanmean(energies);
     accuracies = (snr_mab > SNR_THRESHOLD).astype(float); avg_accuracy = np.nanmean(accuracies);
     avg_snr = np.nanmean(snr_mab);
     return (latency if np.isfinite(latency) else np.nan, avg_throughput if np.isfinite(avg_throughput) else np.nan, avg_energy if np.isfinite(avg_energy) else np.nan, avg_accuracy if np.isfinite(avg_accuracy) else np.nan, avg_snr if np.isfinite(avg_snr) else np.nan)
def train_q_network_per_gru(replay_buffer, q_network, target_network, optimizer, current_beta):
    if len(replay_buffer) < BATCH_SIZE: return 0.0, 0.0
    states, actions, rewards, next_states, batch_indices, is_weights = replay_buffer.sample(BATCH_SIZE, current_beta)
    if not isinstance(states, (list, tuple)) or len(states) < BATCH_SIZE: return 0.0, 0.0
    rewards_scalar = torch.tensor(rewards, dtype=torch.float32).to(device)
    is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32).to(device)
    try:
        states_np = np.stack(states).astype(np.float32); next_states_np = np.stack(next_states).astype(np.float32)
        actions_np = np.stack(actions).astype(np.int64)
        all_states_tensor = torch.from_numpy(states_np).to(device); all_next_states_tensor = torch.from_numpy(next_states_np).to(device); all_actions_tensor = torch.from_numpy(actions_np).to(device)
    except Exception as e:
        print(f"Error converting batch data to tensors: {e}"); return -1.0, -1.0
    batch_size_actual = all_states_tensor.shape[0]; num_ues_actual = all_states_tensor.shape[1]; state_dim = all_states_tensor.shape[2]
    if state_dim != input_size: print(f"ERROR: State dimension mismatch!"); return -1.0, -1.0
    if all_actions_tensor.shape != (batch_size_actual, num_ues_actual): print(f"ERROR: Action dimension mismatch!"); return -1.0, -1.0
    reshaped_states = all_states_tensor.view(batch_size_actual * num_ues_actual, state_dim)
    reshaped_next_states = all_next_states_tensor.view(batch_size_actual * num_ues_actual, state_dim)
    current_q_values_all, _ = q_network(reshaped_states, h_state=None)
    flat_actions = all_actions_tensor.view(-1)
    current_q_values = current_q_values_all.gather(1, flat_actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_q_online_all, _ = q_network(reshaped_next_states, h_state=None); best_next_actions = next_q_online_all.argmax(1)
        next_q_target_all, _ = target_network(reshaped_next_states, h_state=None); max_next_q_values = next_q_target_all.gather(1, best_next_actions.unsqueeze(-1)).squeeze(-1)
    expanded_rewards = rewards_scalar.unsqueeze(1).expand(-1, num_ues_actual).reshape(-1)
    expected_q_values = expanded_rewards + GAMMA * max_next_q_values
    td_errors_elementwise = (current_q_values - expected_q_values).abs()
    td_errors_per_transition = td_errors_elementwise.view(batch_size_actual, num_ues_actual).mean(axis=1)
    td_errors_np = td_errors_per_transition.detach().cpu().numpy()
    replay_buffer.update_priorities(batch_indices, td_errors_np)
    loss_elementwise = nn.MSELoss(reduction='none')(current_q_values, expected_q_values)
    expanded_is_weights = is_weights_tensor.unsqueeze(1).expand(-1, num_ues_actual).reshape(-1); weighted_loss = (expanded_is_weights * loss_elementwise).mean()
    total_batch_loss = weighted_loss.item(); avg_td_error = np.mean(td_errors_np)
    optimizer.zero_grad(); weighted_loss.backward()
    optimizer.step()
    return total_batch_loss, avg_td_error

# --- Function Definition Moved Before Call --- Moved ---
def print_final_results(label, results_agg_dict):
    print(f"{label}:")
    metrics_order = ['latency', 'throughput', 'energy', 'accuracy', 'snr'] # Define order
    results_summary = {} # Store mean/std for bar plot
    for key in metrics_order:
        if key in results_agg_dict:
            values = results_agg_dict[key] # List of average values, one per run
            valid_values = [v for v in values if np.isfinite(v)]
            if len(valid_values) > 0:
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                results_summary[key] = {'mean': mean_val, 'std': std_val} # Store for plot
                if key == 'latency': unit = 'ms'
                elif key == 'throughput': unit = 'Mbps'
                elif key == 'energy': unit = 'mJ'
                elif key == 'snr': unit = 'dB'
                else: unit = '' # For accuracy
                print(f"  Avg {key.capitalize()}: {mean_val:.2f} ± {std_val:.2f} {unit} ({len(valid_values)} valid runs)")
            else:
                print(f"  Avg {key.capitalize()}: No valid data across runs")
                results_summary[key] = {'mean': np.nan, 'std': np.nan}
        else:
            print(f"  Avg {key.capitalize()}: Metric not found")
            results_summary[key] = {'mean': np.nan, 'std': np.nan}
    return results_summary # Return extracted means/stds
# ------------------------------------------------------


# --- Result Aggregation ---
all_runs_results = {'ol': defaultdict(list), 'heuristic': defaultdict(list), 'mab': defaultdict(list)}
all_runs_raw_data = {'ol': defaultdict(list), 'heuristic': defaultdict(list), 'mab': defaultdict(list)}

# --- Main Loop for Multiple Runs ---
start_time_total = time.time()
for run in range(NUM_RUNS):
    run_seed = BASE_SEED + run
    print(f"\n--- Starting Run {run+1}/{NUM_RUNS} (Seed: {run_seed}) ---")
    start_time_run = time.time()

    # Set Seeds
    np.random.seed(run_seed); torch.manual_seed(run_seed)
    try: tf.random.set_seed(run_seed)
    except NameError: pass
    random.seed(run_seed)

    # Re-initialize Models, Optimizer, Buffer, etc.
    q_network = GRUQNetwork(input_size, GRU_HIDDEN_SIZE, NUM_BEAMS).to(device) # Uses new GRU params
    target_network = GRUQNetwork(input_size, GRU_HIDDEN_SIZE, NUM_BEAMS).to(device) # Uses new GRU params
    target_network.load_state_dict(q_network.state_dict()); target_network.eval()
    optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PER_ALPHA)
    current_beta = PER_BETA_START
    mab_counts = np.zeros((NUM_UES, NUM_BEAMS)); mab_values = np.zeros((NUM_UES, NUM_BEAMS))

    # --- Modified Epsilon Initialization ---
    epsilon = EPSILON_START
    # ---------------------------------------
    prev_snr_ol = np.zeros(NUM_UES); prev_is_blocked = np.zeros(NUM_UES, dtype=bool)
    # --- Modified Hidden State Initialization ---
    # Needs shape (num_layers, batch_size, hidden_size) -> (GRU_NUM_LAYERS, NUM_UES, GRU_HIDDEN_SIZE)
    ue_hidden_states = torch.zeros(GRU_NUM_LAYERS, NUM_UES, GRU_HIDDEN_SIZE).to(device)
    # ------------------------------------------
    global_prev_positions = None

    # Initialize channel model
    try:
        channel_model = sn.channel.RayleighBlockFading(num_rx=NUM_UES, num_rx_ant=1, num_tx=1, num_tx_ant=NUM_ANTENNAS)
        print(f"Run {run+1}: Sionna channel model initialized.")
    except Exception as e: print(f"Run {run+1}: Error initializing Sionna channel model: {e}. Exiting."); exit()

    # Training Phase
    print(f"--- Run {run+1}: Starting Training Phase ({NUM_TIMESTEPS} timesteps) ---")
    initial_positions, _ = update_positions(0); initial_h_channel = generate_channel(initial_positions, channel_model)
    if initial_h_channel is not None: prev_snr_ol = initial_beam_scan(initial_h_channel); print(f"Run {run+1}: Initial Avg SNR from Scan: {np.mean(prev_snr_ol):.2f} dB")
    else: print(f"Run {run+1}: Failed to generate initial channel."); prev_snr_ol = np.zeros(NUM_UES)

    for t in range(NUM_TIMESTEPS):
        positions, velocities = update_positions(t); angles = compute_relative_angles(positions);
        norm_distances, actual_distances = compute_distances(positions); h_channel = generate_channel(positions, channel_model);
        norm_velocities = normalize_velocity(velocities);
        if h_channel is None: print(f"Run {run+1} - TS {t+1}: Skipping timestep."); continue

        heuristic_beam_indices = np.zeros(NUM_UES, dtype=int); current_is_blocked = np.zeros(NUM_UES, dtype=bool)
        for i in range(NUM_UES):
            heuristic_beam_indices[i] = np.argmin(np.abs(BEAM_ANGLES - angles[i])); rand_val = np.random.rand()
            if prev_is_blocked[i]: current_is_blocked[i] = (rand_val < P_BB)
            else: current_is_blocked[i] = (rand_val < P_UB)

        norm_prev_snr_ol = normalize_snr(prev_snr_ol); prev_blocked_float = prev_is_blocked.astype(float);
        state_ol = np.array([[ angles[i], norm_prev_snr_ol[i], norm_distances[i], norm_velocities[i], prev_blocked_float[i] ] for i in range(NUM_UES)], dtype=np.float32)
        actions_ol = np.zeros(NUM_UES, dtype=int)
        state_tensor_all_ues = torch.tensor(state_ol, dtype=torch.float32).to(device)

        if np.random.rand() < epsilon:
            actions_ol = np.random.randint(0, NUM_BEAMS, size=NUM_UES)
            with torch.no_grad(): q_network.eval(); _, ue_hidden_states = q_network(state_tensor_all_ues, ue_hidden_states); q_network.train()
        else:
            with torch.no_grad():
                q_network.eval(); q_values_all_ues, ue_hidden_states = q_network(state_tensor_all_ues, ue_hidden_states)
                actions_ol = q_values_all_ues.argmax(dim=1).cpu().numpy(); q_network.train()

        actions_mab = np.zeros(NUM_UES, dtype=int)
        for i in range(NUM_UES): actions_mab[i] = mab_ucb1_action(i, t, mab_counts, mab_values, MAB_EXPLORATION_FACTOR, mab_epsilon)

        snr_ol = np.zeros(NUM_UES); snr_heuristic = np.zeros(NUM_UES); snr_mab = np.zeros(NUM_UES);
        for i in range(NUM_UES):
            action_h_i = heuristic_beam_indices[i]; action_mab_i = actions_mab[i]; action_ol_i = actions_ol[i]
            attenuation_h = BLOCKAGE_ATTENUATION_DB if action_h_i == heuristic_beam_indices[i] and current_is_blocked[i] else 0.0;
            attenuation_mab = BLOCKAGE_ATTENUATION_DB if action_mab_i == heuristic_beam_indices[i] and current_is_blocked[i] else 0.0;
            attenuation_ol = BLOCKAGE_ATTENUATION_DB if action_ol_i == heuristic_beam_indices[i] and current_is_blocked[i] else 0.0;
            snr_heuristic[i] = compute_snr(h_channel, action_h_i, i, extra_attenuation_db=attenuation_h);
            snr_mab[i] = compute_snr(h_channel, action_mab_i, i, extra_attenuation_db=attenuation_mab);
            snr_ol[i] = compute_snr(h_channel, action_ol_i, i, extra_attenuation_db=attenuation_ol);
            reward_mab = snr_mab[i]
            if np.isfinite(reward_mab): mab_counts[i, action_mab_i] += 1; mab_values[i, action_mab_i] += reward_mab

        throughputs_ol = np.array([compute_throughput(snr) for snr in snr_ol]); avg_throughput_ol = np.nanmean(throughputs_ol);
        energies_ol = np.array([compute_energy(snr_ol[i], actual_distances[i]) for i in range(NUM_UES)]); avg_energy_ol = np.nanmean(energies_ol);
        avg_snr_ol = np.nanmean(snr_ol); accuracy_per_ue_ol = (snr_ol > SNR_THRESHOLD).astype(float);
        reward_ol = compute_reward(avg_throughput_ol, avg_snr_ol, np.nanmean(prev_snr_ol), avg_energy_ol, accuracy_per_ue_ol);

        next_positions, next_velocities = update_positions(t + 1);
        next_angles = compute_relative_angles(next_positions); next_norm_distances, _ = compute_distances(next_positions);
        next_norm_velocities = normalize_velocity(next_velocities); norm_current_snr_ol = normalize_snr(snr_ol);
        current_blocked_float = current_is_blocked.astype(float);
        next_state_ol = np.array([[ next_angles[i], norm_current_snr_ol[i], next_norm_distances[i], next_norm_velocities[i], current_blocked_float[i] ] for i in range(NUM_UES)], dtype=np.float32)
        replay_buffer.add(state_ol, actions_ol, reward_ol, next_state_ol)
        prev_snr_ol = snr_ol.copy(); prev_is_blocked = current_is_blocked.copy()

        loss_val = 0.0; td_error_val = 0.0
        if len(replay_buffer) >= BATCH_SIZE:
            loss_val, td_error_val = train_q_network_per_gru(replay_buffer, q_network, target_network, optimizer, current_beta)
            current_beta = min(1.0, current_beta + PER_BETA_INCREMENT)
            if (t + 1) % 500 == 0: print(f"Run {run+1} - TS {t+1}/{NUM_TIMESTEPS}, Eps: {epsilon:.3f}, Beta: {current_beta:.3f}, Loss: {loss_val:.4f}, Avg TD: {td_error_val:.4f}, AvgSNR: {avg_snr_ol:.2f} dB")
        elif (t + 1) % 500 == 0: print(f"Run {run+1} - TS {t+1}/{NUM_TIMESTEPS}, Eps: {epsilon:.3f}, Beta: {current_beta:.3f}, Buffer: {len(replay_buffer)}/{BATCH_SIZE}, AvgSNR: {avg_snr_ol:.2f} dB")

        if (t + 1) % TARGET_UPDATE_FREQ == 0: target_network.load_state_dict(q_network.state_dict())

        # --- Modified Epsilon Decay ---
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        # ----------------------------

    print(f"--- Run {run+1}: Training Phase Completed ---")

    # Evaluation Phase
    print(f"--- Run {run+1}: Starting Evaluation Phase ({EVAL_TIMESTEPS} timesteps) ---")
    results_ol_eval_run = defaultdict(list); results_heuristic_eval_run = defaultdict(list); results_mab_eval_run = defaultdict(list)
    prev_snr_ol_eval = prev_snr_ol.copy(); prev_is_blocked_eval = prev_is_blocked.copy()
    # --- Modified Hidden State Initialization for Eval ---
    eval_ue_hidden_states = ue_hidden_states.clone() # Use final hidden state from training
    # Ensure correct shape if ue_hidden_states wasn't updated in last step (e.g. explore)
    if eval_ue_hidden_states.shape[0] != GRU_NUM_LAYERS:
         eval_ue_hidden_states = torch.zeros(GRU_NUM_LAYERS, NUM_UES, GRU_HIDDEN_SIZE).to(device)
    # -------------------------------------------------
    q_network.eval()

    for t_eval in range(EVAL_TIMESTEPS):
        t = NUM_TIMESTEPS + t_eval;
        positions, velocities = update_positions(t); angles = compute_relative_angles(positions);
        norm_distances, actual_distances = compute_distances(positions); h_channel = generate_channel(positions, channel_model);
        norm_velocities = normalize_velocity(velocities);
        if h_channel is None:
            print(f"Run {run+1} - Eval TS {t_eval+1}: Skipping timestep.");
            for r_dict in [results_ol_eval_run, results_heuristic_eval_run, results_mab_eval_run]:
                for key in ["latency", "throughput", "energy", "accuracy", "snr"]: r_dict[key].append(np.nan)
            continue

        heuristic_beam_indices = np.zeros(NUM_UES, dtype=int); current_is_blocked_eval = np.zeros(NUM_UES, dtype=bool)
        for i in range(NUM_UES):
            heuristic_beam_indices[i] = np.argmin(np.abs(BEAM_ANGLES - angles[i])); rand_val = np.random.rand()
            if prev_is_blocked_eval[i]: current_is_blocked_eval[i] = (rand_val < P_BB)
            else: current_is_blocked_eval[i] = (rand_val < P_UB)

        norm_prev_snr_ol_eval = normalize_snr(prev_snr_ol_eval); prev_blocked_eval_float = prev_is_blocked_eval.astype(float);
        state_ol = np.array([[ angles[i], norm_prev_snr_ol_eval[i], norm_distances[i], norm_velocities[i], prev_blocked_eval_float[i] ] for i in range(NUM_UES)], dtype=np.float32)
        actions_ol = np.zeros(NUM_UES, dtype=int)
        state_tensor_all_ues = torch.tensor(state_ol, dtype=torch.float32).to(device)
        with torch.no_grad():
             q_values_all_ues, eval_ue_hidden_states = q_network(state_tensor_all_ues, eval_ue_hidden_states) # Pass and update hidden state
             actions_ol = q_values_all_ues.argmax(dim=1).cpu().numpy()

        actions_mab = np.zeros(NUM_UES, dtype=int)
        for i in range(NUM_UES):
            counts_i = mab_counts[i, :];
            if np.all(counts_i == 0): action_mab = np.random.randint(NUM_BEAMS)
            else:
                 mean_rewards = np.full(NUM_BEAMS, -np.inf); valid_indices = counts_i > 0;
                 mean_rewards[valid_indices] = np.divide(mab_values[i, valid_indices], counts_i[valid_indices])
                 action_mab = np.random.choice(np.where(mean_rewards == np.max(mean_rewards))[0])
            actions_mab[i] = action_mab

        snr_ol = np.zeros(NUM_UES); snr_heuristic = np.zeros(NUM_UES); snr_mab = np.zeros(NUM_UES);
        for i in range(NUM_UES):
            action_h_i = heuristic_beam_indices[i]; action_mab_i = actions_mab[i]; action_ol_i = actions_ol[i]
            attenuation_h = BLOCKAGE_ATTENUATION_DB if action_h_i == heuristic_beam_indices[i] and current_is_blocked_eval[i] else 0.0;
            attenuation_mab = BLOCKAGE_ATTENUATION_DB if action_mab_i == heuristic_beam_indices[i] and current_is_blocked_eval[i] else 0.0;
            attenuation_ol = BLOCKAGE_ATTENUATION_DB if action_ol_i == heuristic_beam_indices[i] and current_is_blocked_eval[i] else 0.0;
            snr_heuristic[i] = compute_snr(h_channel, action_h_i, i, extra_attenuation_db=attenuation_h);
            snr_mab[i] = compute_snr(h_channel, action_mab_i, i, extra_attenuation_db=attenuation_mab);
            snr_ol[i] = compute_snr(h_channel, action_ol_i, i, extra_attenuation_db=attenuation_ol);

        latency_h, throughput_h, energy_h, accuracy_h, avg_snr_h = angle_heuristic_beam_switching(snr_heuristic, positions);
        results_heuristic_eval_run["latency"].append(latency_h); results_heuristic_eval_run["throughput"].append(throughput_h); results_heuristic_eval_run["energy"].append(energy_h); results_heuristic_eval_run["accuracy"].append(accuracy_h); results_heuristic_eval_run["snr"].append(avg_snr_h);
        latency_mab, throughput_mab, energy_mab, accuracy_mab, avg_snr_mab = mab_ucb1_beam_switching(snr_mab, positions);
        results_mab_eval_run["latency"].append(latency_mab); results_mab_eval_run["throughput"].append(throughput_mab); results_mab_eval_run["energy"].append(energy_mab); results_mab_eval_run["accuracy"].append(accuracy_mab); results_mab_eval_run["snr"].append(avg_snr_mab);
        throughputs_ol = np.array([compute_throughput(snr) for snr in snr_ol]); avg_throughput_ol = np.nanmean(throughputs_ol);
        energies_ol = np.array([compute_energy(snr_ol[i], actual_distances[i]) for i in range(NUM_UES)]); avg_energy_ol = np.nanmean(energies_ol);
        avg_snr_ol = np.nanmean(snr_ol); latency_ol = compute_latency(avg_throughput_ol); accuracy_ol = np.nanmean((snr_ol > SNR_THRESHOLD).astype(float))
        results_ol_eval_run["latency"].append(latency_ol); results_ol_eval_run["throughput"].append(avg_throughput_ol); results_ol_eval_run["energy"].append(avg_energy_ol); results_ol_eval_run["accuracy"].append(accuracy_ol); results_ol_eval_run["snr"].append(avg_snr_ol);

        prev_snr_ol_eval = snr_ol.copy(); prev_is_blocked_eval = current_is_blocked_eval.copy();
        if (t_eval + 1) % 100 == 0: print(f"Run {run+1} - Eval TS {t_eval+1}/{EVAL_TIMESTEPS}, AvgSNR OL: {avg_snr_ol:.2f} dB")

    q_network.train()
    print(f"--- Run {run+1}: Evaluation Phase Completed ---")

    # Aggregate results for this run
    for key in results_ol_eval_run:
        all_runs_results['ol'][key].append(np.nanmean(results_ol_eval_run[key]))
        all_runs_results['heuristic'][key].append(np.nanmean(results_heuristic_eval_run[key]))
        all_runs_results['mab'][key].append(np.nanmean(results_mab_eval_run[key]))
    for key in results_ol_eval_run:
         all_runs_raw_data['ol'][key].append(results_ol_eval_run[key])
         all_runs_raw_data['heuristic'][key].append(results_heuristic_eval_run[key])
         all_runs_raw_data['mab'][key].append(results_mab_eval_run[key])

    end_time_run = time.time()
    print(f"--- Run {run+1} finished in {end_time_run - start_time_run:.2f} seconds ---")

# End of Outer Loop
end_time_total = time.time()
print(f"\n--- All {NUM_RUNS} runs completed in {end_time_total - start_time_total:.2f} seconds ---")

# Save Raw Data
raw_data_filename = "raw_evaluation_data.csv"
print(f"\n--- Saving Raw Evaluation Data to {raw_data_filename} ---")
try:
    with open(raw_data_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['run', 'timestep', 'method', 'metric', 'value']; writer.writerow(header)
        metrics_to_save = ["latency", "throughput", "energy", "accuracy", "snr"]
        for run_idx in range(NUM_RUNS):
            for method in ['ol', 'heuristic', 'mab']:
                for metric in metrics_to_save:
                    if run_idx < len(all_runs_raw_data[method][metric]):
                        raw_values = all_runs_raw_data[method][metric][run_idx]
                        for ts_idx, value in enumerate(raw_values): writer.writerow([run_idx + 1, ts_idx + 1, method, metric, value])
                    else: print(f"Warning: Missing raw data for run {run_idx+1}, method {method}, metric {metric}")
    print(f"Raw data successfully saved.")
except IOError as e: print(f"Error saving raw data to {raw_data_filename}: {e}")
except Exception as e: print(f"An unexpected error occurred while saving raw data: {e}")

# Plot Results of the LAST run
print("\n--- Plotting Results (Last Run - Smoothed Evaluation) ---")
plt.figure(figsize=(15, 10))
metrics = ["latency", "throughput", "energy", "accuracy", "snr"]; ylabels = ["Latency (ms)", "Throughput (Mbps)", "Energy (mJ)", "Accuracy", "Average SNR (dB)"]; subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
title_fontsize=12; title_fontweight='bold'; ylabel_fontweight='bold'; xlabel_fontweight='bold'; xlabel_fontsize=10; legend_fontsize='large'; legend_title_fontsize='large'
color_ol_eval="green"; color_heuristic="orange"; color_mab="red"; eval_plot_range = range(EVAL_TIMESTEPS)
if 'ol' in all_runs_results and all_runs_raw_data['ol']:
    last_run_ol_results = {key: all_runs_raw_data['ol'][key][-1] for key in metrics}
    last_run_heuristic_results = {key: all_runs_raw_data['heuristic'][key][-1] for key in metrics}
    last_run_mab_results = {key: all_runs_raw_data['mab'][key][-1] for key in metrics}
    for i, (key, ylabel) in enumerate(zip(metrics, ylabels), 1):
        plt.subplot(3, 2, i); plot_label_prefix = f"{subplot_labels[i-1]} "
        data_ol_eval = last_run_ol_results.get(key, []); data_heuristic_eval = last_run_heuristic_results.get(key, []); data_mab_eval = last_run_mab_results.get(key, [])
        smoothed_ol_eval = moving_average(np.array(data_ol_eval), SMOOTHING_WINDOW); smoothed_heuristic_eval = moving_average(np.array(data_heuristic_eval), SMOOTHING_WINDOW); smoothed_mab_eval = moving_average(np.array(data_mab_eval), SMOOTHING_WINDOW)
        plot_title = f"{plot_label_prefix}Smoothed {key.capitalize()} Comparison (Evaluation - Last Run)"
        plt.plot(eval_plot_range, smoothed_ol_eval, color=color_ol_eval, linewidth=1.5, label='OL (GRU+PER)' if i==1 else ""); plt.plot(eval_plot_range, smoothed_heuristic_eval, color=color_heuristic, linestyle="--", linewidth=1.0, label='Heuristic' if i==1 else ""); plt.plot(eval_plot_range, smoothed_mab_eval, color=color_mab, linestyle="-.", linewidth=1.0, label='MAB UCB1' if i==1 else "");
        plt.xlabel("Evaluation Timestep", fontweight=xlabel_fontweight, fontsize=xlabel_fontsize); plt.ylabel(ylabel, fontweight=ylabel_fontweight); plt.title(plot_title, fontweight=title_fontweight, fontsize=title_fontsize); plt.grid(True, linestyle=':', alpha=0.6); plt.xlim(0, EVAL_TIMESTEPS)
    ax_legend = plt.subplot(3, 2, 6)
    line_ol_eval = mlines.Line2D([], [], color=color_ol_eval, linestyle='-', linewidth=1.5, label='OL (GRU+PER)'); line_heuristic_eval = mlines.Line2D([], [], color=color_heuristic, linestyle='--', linewidth=1.0, label='Heuristic'); line_mab_eval = mlines.Line2D([], [], color=color_mab, linestyle='-.', linewidth=1.0, label='MAB UCB1')
    ax_legend.legend(handles=[line_ol_eval, line_heuristic_eval, line_mab_eval], loc='center', fontsize=legend_fontsize, frameon=True, title="Methods (Smoothed Evaluation)", title_fontsize=legend_title_fontsize); ax_legend.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    last_run_plot_filename = "results_last_run_gru_per_eval_smoothed.png"
    try: plt.savefig(last_run_plot_filename, dpi=300, bbox_inches='tight'); print(f"Last run results plot saved to {last_run_plot_filename}")
    except Exception as e: print(f"Error saving last run plot: {e}")
    plt.close()
else: print("Skipping last run plot generation as results are not available.")

# --- Print Final Average Metrics Across All Runs --- Moved Definition Before Call ---
print(f"\n--- Average Metrics Across {NUM_RUNS} Runs (Evaluation Phase Only) ---")
final_ol_metrics = print_final_results("OL (GRU+PER) Results", all_runs_results['ol'])
final_heuristic_metrics = print_final_results("Angle Heuristic Results", all_runs_results['heuristic'])
final_mab_metrics = print_final_results("MAB UCB1 Results", all_runs_results['mab'])

# --- Perform t-test ---
print(f"\n--- Statistical Significance (t-test: OL vs MAB, {NUM_RUNS} runs) ---")
ol_throughputs = all_runs_results['ol'].get('throughput', []); mab_throughputs = all_runs_results['mab'].get('throughput', [])
valid_ol_tp = [tp for tp in ol_throughputs if np.isfinite(tp)]; valid_mab_tp = [tp for tp in mab_throughputs if np.isfinite(tp)]
if len(valid_ol_tp) > 1 and len(valid_mab_tp) > 1 and len(valid_ol_tp) == len(valid_mab_tp):
     t_stat_tp, p_value_tp = stats.ttest_ind(valid_ol_tp, valid_mab_tp, equal_var=False, nan_policy='omit')
     print(f"Throughput (OL vs MAB): t-statistic = {t_stat_tp:.3f}, p-value = {p_value_tp:.4f}")
     if p_value_tp < 0.05: print("  Result: Statistically significant difference in Throughput (p < 0.05)")
     else: print("  Result: No statistically significant difference in Throughput (p >= 0.05)")
else: print(f"Throughput (OL vs MAB): Not enough valid comparable data for t-test (OL: {len(valid_ol_tp)}, MAB: {len(valid_mab_tp)}).")

ol_snrs = all_runs_results['ol'].get('snr', []); mab_snrs = all_runs_results['mab'].get('snr', [])
valid_ol_snr = [snr for snr in ol_snrs if np.isfinite(snr)]; valid_mab_snr = [snr for snr in mab_snrs if np.isfinite(snr)]
if len(valid_ol_snr) > 1 and len(valid_mab_snr) > 1 and len(valid_ol_snr) == len(valid_mab_snr):
     t_stat_snr, p_value_snr = stats.ttest_ind(valid_ol_snr, valid_mab_snr, equal_var=False, nan_policy='omit')
     print(f"SNR (OL vs MAB): t-statistic = {t_stat_snr:.3f}, p-value = {p_value_snr:.4f}")
     if p_value_snr < 0.05: print("  Result: Statistically significant difference in SNR (p < 0.05)")
     else: print("  Result: No statistically significant difference in SNR (p >= 0.05)")
else: print(f"SNR (OL vs MAB): Not enough valid comparable data for t-test (OL: {len(valid_ol_snr)}, MAB: {len(valid_mab_snr)}).")

# --- Generate New Bar Plot ---
print("\n--- Plotting Average Performance Bar Chart ---")
bar_plot_filename = "results_average_bar_plot.png"
try:
    labels = ['OL (GRU+PER)', 'Heuristic', 'MAB UCB1']; metrics_to_plot = ['Throughput (Mbps)', 'SNR (dB)']; metric_keys = ['throughput', 'snr']
    means = {key: [] for key in metric_keys}; stds = {key: [] for key in metric_keys}; plot_possible = True
    for method_metrics in [final_ol_metrics, final_heuristic_metrics, final_mab_metrics]:
        for key in metric_keys:
             if key in method_metrics and 'mean' in method_metrics[key] and 'std' in method_metrics[key]:
                 means[key].append(method_metrics[key]['mean']); stds[key].append(method_metrics[key]['std'])
             else: means[key].append(np.nan); stds[key].append(np.nan); plot_possible = False
    if plot_possible and not any(np.isnan(m) for k in means for m in means[k]):
        x = np.arange(len(labels)); width = 0.35; fig_bar, axes = plt.subplots(1, 2, figsize=(12, 5)); colors = ['green', 'orange', 'red']
        for i, (metric_label, key) in enumerate(zip(metrics_to_plot, metric_keys)):
            ax = axes[i]; error_values = [max(0, s) for s in stds[key]]
            rects = ax.bar(x, means[key], width, yerr=error_values, label=labels, color=colors, capsize=5)
            ax.set_ylabel(metric_label, fontweight=ylabel_fontweight); ax.set_title(f'Average {metric_label} Comparison ({NUM_RUNS} Runs)', fontweight=title_fontweight)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right"); ax.grid(True, linestyle=':', alpha=0.6, axis='y')
            ax.bar_label(rects, fmt='%.2f', padding=3)
        fig_bar.tight_layout(); plt.savefig(bar_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Average results bar plot saved to {bar_plot_filename}")
        plt.close(fig_bar)
    else: print("Skipping average bar plot generation due to missing or invalid data.")
except Exception as e: print(f"Error generating average results bar plot: {e}")

print("\n--- Script Finished ---")