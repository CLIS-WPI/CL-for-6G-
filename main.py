import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
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
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0]}")
    else:
        print("No GPU found by TensorFlow.")
except Exception as e:
    print(f"Error configuring TensorFlow GPU: {e}")
    print("TensorFlow might not be using the GPU correctly.")


# Select device dynamically for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# Simulation Parameters
NUM_UES = 5
NUM_ANTENNAS = 64
NUM_BEAMS = NUM_ANTENNAS
ROAD_LENGTH = 500
BS_POSITION = ROAD_LENGTH / 2
FREQ = 28e9
TX_POWER_DBM = 30
NOISE_POWER_DBM = -70 # Keep the more challenging noise level
NUM_TIMESTEPS = 4000
EVAL_TIMESTEPS = 400
TIMESTEP_DURATION = 0.01
BUFFER_SIZE = 10000
BATCH_SIZE = 64
LEARNING_RATE = 0.0001 # Keep reduced LR
GAMMA = 0.99
SNR_THRESHOLD = 14.0
TARGET_UPDATE_FREQ = 50
PATH_LOSS_EXPONENT = 2.5
MAB_EXPLORATION_FACTOR = 2.0
VELOCITY_NORMALIZATION_FACTOR = 20.0 # Approx max speed in m/s
SNR_NORM_MIN = -10.0 # Min expected SNR for normalization
SNR_NORM_MAX = 50.0  # Max expected SNR for normalization

# --- Time-Correlated Blockage Model Parameters ---
BLOCKAGE_ATTENUATION_DB = 25.0 # Keep attenuation level
P_BB = 0.85 # Probability Blocked -> Blocked
P_UB = 0.03 # Probability Unblocked -> Blocked
# ---------------------------------------------

BANDWIDTH = 100e6

# Generate DFT-based codebook
def generate_codebook(num_antennas, num_beams):
    if num_beams != num_antennas:
        print(f"Warning: num_beams ({num_beams}) != num_antennas ({num_antennas}). Adjusting num_beams.")
        num_beams = num_antennas
    angles = np.linspace(-np.pi/2, np.pi/2, num_beams)
    codebook = np.zeros((num_antennas, num_beams), dtype=complex)
    antenna_indices = np.arange(num_antennas)
    for i, theta in enumerate(angles):
        steering_vector = np.exp(1j * np.pi * antenna_indices * np.sin(theta))
        codebook[:, i] = steering_vector / np.sqrt(num_antennas)
    return codebook

CODEBOOK = generate_codebook(NUM_ANTENNAS, NUM_BEAMS)
BEAM_ANGLES = np.linspace(-np.pi/2, np.pi/2, NUM_BEAMS) # Store beam angles globally

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        # Adjust hidden layers if needed for larger input
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # --- Add input normalization if not done before passing ---
        # Example: x = (x - mean) / std
        # ---
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# --- Initialize models for DQL with NEW input size ---
input_size = 5  # Angle, Norm_SNR, Norm_Dist, Norm_Velocity, Prev_Block_Status
q_network = QNetwork(input_size, NUM_BEAMS).to(device)
target_network = QNetwork(input_size, NUM_BEAMS).to(device)
target_network.load_state_dict(q_network.state_dict()) # Initialize target same as online
optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
replay_buffer = deque(maxlen=BUFFER_SIZE)
# -----------------------------------------------------

# MAB UCB1 Initialization
mab_counts = np.zeros((NUM_UES, NUM_BEAMS))
mab_values = np.zeros((NUM_UES, NUM_BEAMS))
mab_epsilon = 1e-6

# Channel configuration
try:
    channel_model = sn.channel.RayleighBlockFading(
         num_rx=NUM_UES, num_rx_ant=1, num_tx=1, num_tx_ant=NUM_ANTENNAS
    )
except AttributeError:
     print("Error: Could not find sn.channel.RayleighBlockFading.")
     exit()

# Path Loss Calculation
def compute_path_loss(distances):
    min_distance = 1.0 # Minimum distance clamp
    distances = np.maximum(distances, min_distance)
    path_loss_db = 32.45 + 10 * PATH_LOSS_EXPONENT * np.log10(distances + 1e-9)
    path_loss_linear = 10 ** (-path_loss_db / 10)
    return path_loss_linear, path_loss_db

# Compute SNR (applies extra attenuation)
def compute_snr(h_channel_all_ues, beam_idx, ue_idx, extra_attenuation_db=0.0):
    h_ue = h_channel_all_ues[ue_idx, :]
    beam = CODEBOOK[:, beam_idx]
    effective_signal_amplitude_sq = np.abs(np.dot(np.conj(h_ue), beam))**2
    noise_variance_relative = 10**((NOISE_POWER_DBM - TX_POWER_DBM) / 10)
    snr_linear = effective_signal_amplitude_sq / (noise_variance_relative + 1e-15)
    snr_db = 10 * np.log10(snr_linear + 1e-15)
    snr_db -= extra_attenuation_db # Apply blockage attenuation
    snr_db = np.clip(snr_db, -50, 50) # Keep clipping
    return snr_db

# Channel Generation
def generate_channel(positions):
    try:
        h_tuple = channel_model(batch_size=1, num_time_steps=1)
        h = h_tuple[0].numpy()
    except Exception as e:
         print(f"Error during Sionna channel model execution: {e}")
         print("Falling back to simple Gaussian channel model.")
         h = (np.random.randn(1, NUM_UES, 1, 1, NUM_ANTENNAS, 1) +
              1j * np.random.randn(1, NUM_UES, 1, 1, NUM_ANTENNAS, 1)) / np.sqrt(2)
    h = h.reshape(NUM_UES, NUM_ANTENNAS)
    distances = np.abs(positions - BS_POSITION)
    path_loss_linear, path_loss_db = compute_path_loss(distances)
    h_channel = h * np.sqrt(path_loss_linear[:, np.newaxis])
    return h_channel

# Initial Beam Scan
def initial_beam_scan(h_channel):
    initial_snr = np.zeros(NUM_UES)
    for i in range(NUM_UES):
        best_snr = -float('inf')
        for beam_idx in range(NUM_BEAMS):
            snr = compute_snr(h_channel, beam_idx, i, extra_attenuation_db=0.0)
            if snr > best_snr: best_snr = snr
        initial_snr[i] = best_snr
    return initial_snr

# --- MODIFIED UE Position Update to return velocity ---
# Store previous positions globally or pass them around
# Global storage is simpler for this script structure
global_prev_positions = None

def update_positions(t):
    global global_prev_positions
    positions = np.zeros(NUM_UES)
    velocities = np.zeros(NUM_UES) # Speed along the road axis

    # Calculate current positions
    for i in range(NUM_UES):
         freq = 0.01 + i * 0.005
         movement_range = 200
         positions[i] = BS_POSITION + movement_range * np.sin(freq * t * TIMESTEP_DURATION)
         positions[i] = np.clip(positions[i], 0, ROAD_LENGTH)

    # Calculate velocities (handle first step)
    if global_prev_positions is not None:
        velocities = (positions - global_prev_positions) / TIMESTEP_DURATION
    # else: velocities remain zero for t=0

    # Update previous positions for next step
    global_prev_positions = positions.copy()

    return positions, velocities # Return both
# ----------------------------------------------------

# Utility Functions
def compute_relative_angles(positions):
    return np.arctan2(positions - BS_POSITION, 10)

def compute_distances(positions):
    distances = np.abs(positions - BS_POSITION)
    normalized_distances = distances / ROAD_LENGTH
    return normalized_distances, distances

# --- Utility Functions for State Normalization ---
def normalize_snr(snr_db):
    """Normalizes SNR to roughly [0, 1] range."""
    return np.clip((snr_db - SNR_NORM_MIN) / (SNR_NORM_MAX - SNR_NORM_MIN), 0.0, 1.0)

def normalize_velocity(velocity_mps):
    """Normalizes velocity to roughly [-1, 1] range."""
    return np.clip(velocity_mps / VELOCITY_NORMALIZATION_FACTOR, -1.0, 1.0)
# -------------------------------------------------

# Compute Reward (Absolute Performance based)
def compute_reward(throughput, snr, prev_snr, energy, accuracy_per_ue):
     w_tput = 1.0; w_stab = 0.2; w_energy = -0.3; w_acc = 0.5
     throughput_reward = w_tput * (throughput / 1500.0)
     stability_bonus = w_stab / (1 + abs(snr - prev_snr) + 1e-6)
     energy_penalty = w_energy * (max(0, energy) / 10)
     accuracy_reward = w_acc * np.mean(accuracy_per_ue)
     reward = throughput_reward + stability_bonus + energy_penalty + accuracy_reward
     return reward

def compute_throughput(snr_db):
    snr_linear = 10**(snr_db / 10)
    if snr_linear > 1e15: snr_linear = 1e15
    throughput_bps = BANDWIDTH * np.log2(1 + snr_linear)
    return throughput_bps / 1e6

def compute_latency(avg_throughput_mbps):
    base_latency = 0.5; max_additional_latency = 5
    exp_term = np.clip((avg_throughput_mbps - 400) / 100, -50, 50)
    latency = base_latency + max_additional_latency / (1 + np.exp(exp_term))
    return latency

def compute_energy(snr_db, distance_actual):
    base_energy_mj = 3.0; snr_factor = 0.1 * max(0, snr_db); distance_factor = 0.01 * distance_actual
    energy = base_energy_mj + snr_factor + distance_factor + 0.05 * max(0, snr_db) * (distance_actual / 100)
    return max(0.1, energy)

# DQL Training Function (Handles new state dimension)
def train_q_network():
    if len(replay_buffer) < BATCH_SIZE: return 0.0
    batch_indices = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
    batch_data = [replay_buffer[i] for i in batch_indices]
    states, actions, rewards, next_states = zip(*batch_data)
    rewards_scalar = torch.tensor([r for r in rewards], dtype=torch.float32).to(device)
    states_np = np.array(states, dtype=np.float32); next_states_np = np.array(next_states, dtype=np.float32)
    actions_np = np.array(actions, dtype=np.int64)
    all_states_tensor = torch.from_numpy(states_np).to(device); all_actions_tensor = torch.from_numpy(actions_np).to(device)
    all_next_states_tensor = torch.from_numpy(next_states_np).to(device)
    batch_size_actual = all_states_tensor.shape[0]; num_ues_actual = all_states_tensor.shape[1]
    state_dim = all_states_tensor.shape[2] # Should be 5 now
    # print(f"DEBUG: state_dim in train_q_network: {state_dim}") # Optional debug
    if state_dim != input_size:
        print(f"ERROR: State dimension mismatch! Expected {input_size}, Got {state_dim}")
        # Handle error appropriately, maybe skip training this batch
        return -1.0 # Indicate error

    reshaped_states = all_states_tensor.view(batch_size_actual * num_ues_actual, state_dim)
    reshaped_next_states = all_next_states_tensor.view(batch_size_actual * num_ues_actual, state_dim)
    q_values_all = q_network(reshaped_states); flat_actions = all_actions_tensor.view(-1)
    current_q_values = q_values_all.gather(1, flat_actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_q_values_all = target_network(reshaped_next_states); max_next_q_values = next_q_values_all.max(1)[0]
    expanded_rewards = rewards_scalar.unsqueeze(1).expand(-1, num_ues_actual).reshape(-1)
    expected_q_values = expanded_rewards + GAMMA * max_next_q_values
    loss = nn.MSELoss()(current_q_values, expected_q_values)
    total_batch_loss = loss.item()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return total_batch_loss

# Angle Heuristic Baseline (Calculates metrics from pre-calculated SNR)
def angle_heuristic_beam_switching(snr_heuristic, positions_actual):
    distances_norm, distances_actual = compute_distances(positions_actual)
    throughputs_heuristic = np.array([compute_throughput(snr) for snr in snr_heuristic])
    avg_throughput_heuristic = np.mean(throughputs_heuristic)
    latency_heuristic = compute_latency(avg_throughput_heuristic)
    energies_heuristic = np.array([compute_energy(snr_heuristic[i], distances_actual[i]) for i in range(NUM_UES)])
    avg_energy_heuristic = np.mean(energies_heuristic)
    accuracies_heuristic = (snr_heuristic > SNR_THRESHOLD).astype(float)
    avg_accuracy_heuristic = np.mean(accuracies_heuristic)
    return latency_heuristic, avg_throughput_heuristic, avg_energy_heuristic, avg_accuracy_heuristic, np.mean(snr_heuristic)

# MAB UCB1 Action Selection
def mab_ucb1_action(ue_idx, t, mab_counts, mab_values, exploration_factor=2.0):
    unexplored_arms = np.where(mab_counts[ue_idx, :] == 0)[0]
    if len(unexplored_arms) > 0: return unexplored_arms[0]
    total_counts_ue = np.sum(mab_counts[ue_idx, :]); ucb_values = np.zeros(NUM_BEAMS)
    for beam_idx in range(NUM_BEAMS):
         count = mab_counts[ue_idx, beam_idx]; mean_reward = mab_values[ue_idx, beam_idx] / count
         exploration_bonus = np.sqrt(exploration_factor * np.log(max(1, total_counts_ue)) / (count + mab_epsilon))
         ucb_values[beam_idx] = mean_reward + exploration_bonus
    return np.argmax(ucb_values)

# MAB UCB1 Baseline Logic (Calculates metrics from pre-calculated SNR)
def mab_ucb1_beam_switching(snr_mab, positions_actual):
    distances_norm, distances_actual = compute_distances(positions_actual)
    throughputs_mab = np.array([compute_throughput(snr) for snr in snr_mab])
    avg_throughput_mab = np.mean(throughputs_mab)
    latency_mab = compute_latency(avg_throughput_mab)
    energies_mab = np.array([compute_energy(snr_mab[i], distances_actual[i]) for i in range(NUM_UES)])
    avg_energy_mab = np.mean(energies_mab)
    accuracies_mab = (snr_mab > SNR_THRESHOLD).astype(float)
    avg_accuracy_mab = np.mean(accuracies_mab)
    return latency_mab, avg_throughput_mab, avg_energy_mab, avg_accuracy_mab, np.mean(snr_mab)


# --- Training Phase ---
print("--- Starting Training Phase ---")
results_cl = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
results_angle_heuristic = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
results_mab_ucb = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
snr_log_cl = []; snr_log_angle_heuristic = []; snr_log_mab_ucb = []
epsilon = 1.0
prev_snr_cl = np.zeros(NUM_UES)
prev_is_blocked = np.zeros(NUM_UES, dtype=bool) # Blockage state from previous step
prev_positions = None # Store previous positions for velocity calc

initial_positions, _ = update_positions(0) # Get initial positions
initial_h_channel = generate_channel(initial_positions)
prev_snr_cl = initial_beam_scan(initial_h_channel)
print(f"Initial Avg SNR from Scan: {np.mean(prev_snr_cl):.2f} dB")
# Initialize prev_positions after the first call to update_positions
prev_positions = initial_positions.copy()

for t in range(NUM_TIMESTEPS):
    # 1. Environment Update
    positions, velocities = update_positions(t) # Get current positions and velocities
    angles = compute_relative_angles(positions)
    norm_distances, actual_distances = compute_distances(positions)
    h_channel = generate_channel(positions)
    norm_velocities = normalize_velocity(velocities) # Normalize velocities

    # --- Time-Correlated Blockage Calculation ---
    heuristic_beam_indices = np.zeros(NUM_UES, dtype=int)
    for i in range(NUM_UES): heuristic_beam_indices[i] = np.argmin(np.abs(BEAM_ANGLES - angles[i]))
    current_is_blocked = np.zeros(NUM_UES, dtype=bool)
    for i in range(NUM_UES):
        rand_val = np.random.rand()
        if prev_is_blocked[i]: current_is_blocked[i] = (rand_val < P_BB)
        else: current_is_blocked[i] = (rand_val < P_UB)
    # ---------------------------------------------

    # --- Calculate SNRs for ALL methods considering blockage ---
    snr_cl = np.zeros(NUM_UES); snr_heuristic = np.zeros(NUM_UES); snr_mab = np.zeros(NUM_UES)
    actions_cl = np.zeros(NUM_UES, dtype=int); actions_mab = np.zeros(NUM_UES, dtype=int)

    # --- Construct NEW DRL State (5 dimensions) ---
    norm_prev_snr_cl = normalize_snr(prev_snr_cl) # Normalize previous SNR
    prev_blocked_float = prev_is_blocked.astype(float) # Convert previous blockage to float
    # Note: Velocity calculated based on t-1 and t, so it's info available at step t
    # Use velocity calculated *before* this loop iteration based on pos[t] and pos[t-1]
    state_cl = np.array([[
        angles[i],
        norm_prev_snr_cl[i],
        norm_distances[i],
        norm_velocities[i], # Use normalized velocity
        prev_blocked_float[i] # Use previous blockage status
    ] for i in range(NUM_UES)])
    # ---------------------------------------------

    for i in range(NUM_UES):
        action_h = heuristic_beam_indices[i]
        actions_mab[i] = mab_ucb1_action(i, t, mab_counts, mab_values, MAB_EXPLORATION_FACTOR)
        action_mab = actions_mab[i]
        if np.random.rand() < epsilon: actions_cl[i] = np.random.randint(NUM_BEAMS)
        else:
            with torch.no_grad():
                # Ensure state_cl[i] has 5 elements before converting
                state_tensor_i = torch.tensor(state_cl[i], dtype=torch.float32).unsqueeze(0).to(device)
                actions_cl[i] = q_network(state_tensor_i).argmax().item()
        action_cl = actions_cl[i]

        # Determine attenuation based on CURRENT blockage state
        attenuation_h = BLOCKAGE_ATTENUATION_DB if action_h == heuristic_beam_indices[i] and current_is_blocked[i] else 0.0
        attenuation_mab = BLOCKAGE_ATTENUATION_DB if action_mab == heuristic_beam_indices[i] and current_is_blocked[i] else 0.0
        attenuation_cl = BLOCKAGE_ATTENUATION_DB if action_cl == heuristic_beam_indices[i] and current_is_blocked[i] else 0.0

        # Compute SNRs
        snr_heuristic[i] = compute_snr(h_channel, action_h, i, extra_attenuation_db=attenuation_h)
        snr_mab[i] = compute_snr(h_channel, action_mab, i, extra_attenuation_db=attenuation_mab)
        snr_cl[i] = compute_snr(h_channel, action_cl, i, extra_attenuation_db=attenuation_cl)

        # --- MAB Update ---
        reward_mab = snr_mab[i]
        mab_counts[i, action_mab] += 1
        mab_values[i, action_mab] += reward_mab
        # ------------------

    # --- Calculate Metrics and Rewards ---
    latency_h, throughput_h, energy_h, accuracy_h, avg_snr_h = angle_heuristic_beam_switching(snr_heuristic, positions)
    results_angle_heuristic["latency"].append(latency_h); results_angle_heuristic["throughput"].append(throughput_h)
    results_angle_heuristic["energy"].append(energy_h); results_angle_heuristic["accuracy"].append(accuracy_h)
    snr_log_angle_heuristic.append(avg_snr_h)

    latency_mab, throughput_mab, energy_mab, accuracy_mab, avg_snr_mab = mab_ucb1_beam_switching(snr_mab, positions)
    results_mab_ucb["latency"].append(latency_mab); results_mab_ucb["throughput"].append(throughput_mab)
    results_mab_ucb["energy"].append(energy_mab); results_mab_ucb["accuracy"].append(accuracy_mab)
    snr_log_mab_ucb.append(avg_snr_mab)

    throughputs_cl = np.array([compute_throughput(snr) for snr in snr_cl]); avg_throughput_cl = np.mean(throughputs_cl)
    energies_cl = np.array([compute_energy(snr_cl[i], actual_distances[i]) for i in range(NUM_UES)]); avg_energy_cl = np.mean(energies_cl)
    avg_snr_cl = np.mean(snr_cl); latency_cl = compute_latency(avg_throughput_cl)
    accuracy_per_ue = (snr_cl > SNR_THRESHOLD).astype(float); avg_accuracy_cl = np.mean(accuracy_per_ue)
    reward_cl = compute_reward(avg_throughput_cl, avg_snr_cl, np.mean(prev_snr_cl), avg_energy_cl, accuracy_per_ue)
    results_cl["latency"].append(latency_cl); results_cl["throughput"].append(avg_throughput_cl)
    results_cl["energy"].append(avg_energy_cl); results_cl["accuracy"].append(avg_accuracy_cl)
    snr_log_cl.append(avg_snr_cl)

    # --- DQL Experience Replay ---
    # Need to construct the NEXT state with 5 dimensions
    next_positions, next_velocities = update_positions(t + 1) # Need velocity for next state
    next_angles = compute_relative_angles(next_positions)
    next_norm_distances, _ = compute_distances(next_positions)
    next_norm_velocities = normalize_velocity(next_velocities)
    norm_current_snr_cl = normalize_snr(snr_cl) # Use current SNR as "prev_snr" for next state
    current_blocked_float = current_is_blocked.astype(float) # Use current blockage as "prev_block" for next state

    next_state_cl = np.array([[
        next_angles[i],
        norm_current_snr_cl[i],
        next_norm_distances[i],
        next_norm_velocities[i],
        current_blocked_float[i]
    ] for i in range(NUM_UES)])

    replay_buffer.append((state_cl, actions_cl, reward_cl, next_state_cl)) # Store 5D states
    prev_snr_cl = snr_cl.copy() # Keep unnormalized SNR for reward calculation

    # --- Update Blockage and Position State for Next Timestep ---
    prev_is_blocked = current_is_blocked.copy()
    # prev_positions is updated inside update_positions now
    # ---------------------------------------------

    # Train DQL Network
    loss_val = train_q_network()

    # Update Target Network and Epsilon
    if (t + 1) % TARGET_UPDATE_FREQ == 0: target_network.load_state_dict(q_network.state_dict())
    epsilon = max(0.1, epsilon * 0.999)

    if (t + 1) % 200 == 0:
        print(f"Training Timestep: {t+1}/{NUM_TIMESTEPS}, Epsilon: {epsilon:.3f}, AvgLoss: {loss_val:.4f}, AvgSNR CL: {avg_snr_cl:.2f} dB")

print("--- Training Phase Completed ---")


# --- Evaluation Phase ---
print("--- Starting Evaluation Phase ---")
results_cl_eval = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
results_angle_heuristic_eval = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
results_mab_ucb_eval = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
snr_log_cl_eval = []; snr_log_angle_heuristic_eval = []; snr_log_mab_ucb_eval = []
prev_snr_cl_eval = prev_snr_cl.copy()
prev_is_blocked_eval = prev_is_blocked.copy() # Carry over last blockage state? Or reset? Let's reset.
prev_is_blocked_eval = np.zeros(NUM_UES, dtype=bool)
# Need previous positions for eval velocity calculation
prev_positions_eval = global_prev_positions.copy() # Use the last positions from training

for t_eval in range(EVAL_TIMESTEPS):
    t = NUM_TIMESTEPS + t_eval

    # 1. Environment Update
    positions, velocities = update_positions(t) # Get current positions and velocities
    angles = compute_relative_angles(positions)
    norm_distances, actual_distances = compute_distances(positions)
    h_channel = generate_channel(positions)
    norm_velocities = normalize_velocity(velocities) # Normalize velocities

    # --- Time-Correlated Blockage Calculation (Eval) ---
    heuristic_beam_indices = np.zeros(NUM_UES, dtype=int)
    for i in range(NUM_UES): heuristic_beam_indices[i] = np.argmin(np.abs(BEAM_ANGLES - angles[i]))
    current_is_blocked_eval = np.zeros(NUM_UES, dtype=bool)
    for i in range(NUM_UES):
        rand_val = np.random.rand()
        if prev_is_blocked_eval[i]: current_is_blocked_eval[i] = (rand_val < P_BB)
        else: current_is_blocked_eval[i] = (rand_val < P_UB)
    # ---------------------------------------------------

    # --- Calculate SNRs for ALL methods considering blockage (Eval) ---
    snr_cl = np.zeros(NUM_UES); snr_heuristic = np.zeros(NUM_UES); snr_mab = np.zeros(NUM_UES)
    actions_cl = np.zeros(NUM_UES, dtype=int); actions_mab = np.zeros(NUM_UES, dtype=int)

    # --- Construct DRL State for Eval (5 dimensions) ---
    norm_prev_snr_cl_eval = normalize_snr(prev_snr_cl_eval)
    prev_blocked_eval_float = prev_is_blocked_eval.astype(float)
    # Use current velocity and previous blockage for current state
    state_cl = np.array([[
        angles[i],
        norm_prev_snr_cl_eval[i],
        norm_distances[i],
        norm_velocities[i],
        prev_blocked_eval_float[i]
    ] for i in range(NUM_UES)])
    # ----------------------------------------------------

    for i in range(NUM_UES):
        action_h = heuristic_beam_indices[i]
        # MAB (Exploitation)
        counts_i = mab_counts[i, :];
        if np.all(counts_i == 0): actions_mab[i] = np.random.randint(NUM_BEAMS)
        else: mean_rewards = np.full(NUM_BEAMS, -np.inf); valid_indices = counts_i > 0
        mean_rewards[valid_indices] = np.divide(mab_values[i, valid_indices], counts_i[valid_indices]); actions_mab[i] = np.argmax(mean_rewards)
        action_mab = actions_mab[i]
        # DQL (Exploitation)
        with torch.no_grad():
             state_tensor_i = torch.tensor(state_cl[i], dtype=torch.float32).unsqueeze(0).to(device)
             actions_cl[i] = q_network(state_tensor_i).argmax().item()
        action_cl = actions_cl[i]

        # Determine attenuation based on CURRENT eval blockage state
        attenuation_h = BLOCKAGE_ATTENUATION_DB if action_h == heuristic_beam_indices[i] and current_is_blocked_eval[i] else 0.0
        attenuation_mab = BLOCKAGE_ATTENUATION_DB if action_mab == heuristic_beam_indices[i] and current_is_blocked_eval[i] else 0.0
        attenuation_cl = BLOCKAGE_ATTENUATION_DB if action_cl == heuristic_beam_indices[i] and current_is_blocked_eval[i] else 0.0

        # Compute SNRs
        snr_heuristic[i] = compute_snr(h_channel, action_h, i, extra_attenuation_db=attenuation_h)
        snr_mab[i] = compute_snr(h_channel, action_mab, i, extra_attenuation_db=attenuation_mab)
        snr_cl[i] = compute_snr(h_channel, action_cl, i, extra_attenuation_db=attenuation_cl)

    # --- Calculate Metrics (Eval) ---
    latency_h, throughput_h, energy_h, accuracy_h, avg_snr_h = angle_heuristic_beam_switching(snr_heuristic, positions)
    results_angle_heuristic_eval["latency"].append(latency_h); results_angle_heuristic_eval["throughput"].append(throughput_h)
    results_angle_heuristic_eval["energy"].append(energy_h); results_angle_heuristic_eval["accuracy"].append(accuracy_h)
    snr_log_angle_heuristic_eval.append(avg_snr_h)

    latency_mab, throughput_mab, energy_mab, accuracy_mab, avg_snr_mab = mab_ucb1_beam_switching(snr_mab, positions)
    results_mab_ucb_eval["latency"].append(latency_mab); results_mab_ucb_eval["throughput"].append(throughput_mab)
    results_mab_ucb_eval["energy"].append(energy_mab); results_mab_ucb_eval["accuracy"].append(accuracy_mab)
    snr_log_mab_ucb_eval.append(avg_snr_mab)

    throughputs_cl = np.array([compute_throughput(snr) for snr in snr_cl]); avg_throughput_cl = np.mean(throughputs_cl)
    energies_cl = np.array([compute_energy(snr_cl[i], actual_distances[i]) for i in range(NUM_UES)]); avg_energy_cl = np.mean(energies_cl)
    avg_snr_cl = np.mean(snr_cl); latency_cl = compute_latency(avg_throughput_cl)
    accuracy_cl = np.mean((snr_cl > SNR_THRESHOLD).astype(float))
    results_cl_eval["latency"].append(latency_cl); results_cl_eval["throughput"].append(avg_throughput_cl)
    results_cl_eval["energy"].append(avg_energy_cl); results_cl_eval["accuracy"].append(accuracy_cl)
    snr_log_cl_eval.append(avg_snr_cl)

    # --- Update Blockage and Position State for Next Eval Timestep ---
    prev_is_blocked_eval = current_is_blocked_eval.copy()
    prev_positions_eval = positions.copy() # Update previous positions for next velocity calc
    # --------------------------------------------------
    prev_snr_cl_eval = snr_cl.copy()

    if (t_eval + 1) % 100 == 0:
         print(f"Evaluation Timestep: {t_eval+1}/{EVAL_TIMESTEPS}, AvgSNR CL: {avg_snr_cl:.2f} dB")

print("--- Evaluation Phase Completed ---")

# --- Plot Results ---
print("--- Plotting Results ---")
plt.figure(figsize=(15, 12))
metrics = ["latency", "throughput", "energy", "accuracy"]
ylabels = ["Latency (ms)", "Throughput (Mbps)", "Energy (mJ)", "Accuracy"]
color_cl_train = "blue"; color_cl_eval = "green"; color_heuristic = "orange"; color_mab = "red"
for i, (key, ylabel) in enumerate(zip(metrics, ylabels), 1):
    plt.subplot(3, 2, i)
    plt.plot(results_cl[key], label=f"CL ({key.capitalize()}) Train", color=color_cl_train, linewidth=1.5)
    plt.plot(results_angle_heuristic[key], label=f"Heuristic ({key.capitalize()})", color=color_heuristic, linestyle="--", linewidth=1.0)
    plt.plot(results_mab_ucb[key], label=f"MAB UCB1 ({key.capitalize()}) Train", color=color_mab, linestyle="-.", linewidth=1.0)
    eval_range = range(NUM_TIMESTEPS, NUM_TIMESTEPS + EVAL_TIMESTEPS)
    plt.plot(eval_range, results_cl_eval[key], label=f"CL ({key.capitalize()}) Eval", color=color_cl_eval, linewidth=1.5)
    plt.plot(eval_range, results_angle_heuristic_eval[key], linestyle="--", color=color_heuristic, linewidth=1.0)
    plt.plot(eval_range, results_mab_ucb_eval[key], label=f"MAB UCB1 ({key.capitalize()}) Eval", color=color_mab, linestyle=":", linewidth=1.5)
    plt.xlabel("Timestep"); plt.ylabel(ylabel); plt.title(f"Comparison of {key.capitalize()}")
    plt.legend(fontsize='small'); plt.grid(True, linestyle=':', alpha=0.6)
# Plot SNR
plt.subplot(3, 2, 5)
plt.plot(snr_log_cl, label="CL Avg SNR (Train)", color=color_cl_train, linewidth=1.5)
plt.plot(snr_log_angle_heuristic, label="Heuristic Avg SNR", color=color_heuristic, linestyle="--", linewidth=1.0)
plt.plot(snr_log_mab_ucb, label="MAB UCB1 Avg SNR (Train)", color=color_mab, linestyle="-.", linewidth=1.0)
eval_range = range(NUM_TIMESTEPS, NUM_TIMESTEPS + EVAL_TIMESTEPS)
plt.plot(eval_range, snr_log_cl_eval, label="CL Avg SNR (Eval)", color=color_cl_eval, linewidth=1.5)
plt.plot(eval_range, snr_log_angle_heuristic_eval, linestyle="--", color=color_heuristic, linewidth=1.0)
plt.plot(eval_range, snr_log_mab_ucb_eval, label="MAB UCB1 Avg SNR (Eval)", color=color_mab, linestyle=":", linewidth=1.5)
plt.xlabel("Timestep"); plt.ylabel("Average SNR (dB)"); plt.title("Average SNR Comparison")
plt.legend(fontsize='small'); plt.grid(True, linestyle=':', alpha=0.6)
plt.suptitle("Online Learning Beam Switching Performance Comparison (Correlated Blockage, Enhanced State)", fontsize=16, y=1.02) # Updated title
plt.tight_layout(rect=[0, 0, 1, 1])
try:
    plt.savefig("results_comparison_enhanced_state.png", dpi=300, bbox_inches='tight') # Updated filename
    print("Results plot saved to results_comparison_enhanced_state.png")
except Exception as e: print(f"Error saving plot: {e}")
# plt.show()

# --- Print average metrics ---
print("\n--- Average Metrics ---")
def print_avg_results(label, results_dict, snr_log):
    if not results_dict['latency'] or not snr_log: print(f"{label}: No data to report."); return
    print(f"{label}:")
    print(f"  Avg Latency:    {np.mean(results_dict['latency']):.2f} ms")
    print(f"  Avg Throughput: {np.mean(results_dict['throughput']):.2f} Mbps")
    print(f"  Avg Energy:     {np.mean(results_dict['energy']):.2f} mJ")
    print(f"  Avg Accuracy:   {np.mean(results_dict['accuracy']):.2f}")
    print(f"  Avg SNR:        {np.mean(snr_log):.2f} dB")

print_avg_results("CL Results (Training)", results_cl, snr_log_cl)
print_avg_results("CL Results (Evaluation)", results_cl_eval, snr_log_cl_eval)
print_avg_results("Angle Heuristic Results (Training)", results_angle_heuristic, snr_log_angle_heuristic)
print_avg_results("Angle Heuristic Results (Evaluation)", results_angle_heuristic_eval, snr_log_angle_heuristic_eval)
print_avg_results("MAB UCB1 Results (Training)", results_mab_ucb, snr_log_mab_ucb)
print_avg_results("MAB UCB1 Results (Evaluation)", results_mab_ucb_eval, snr_log_mab_ucb_eval)

print("\n--- Script Finished ---")
