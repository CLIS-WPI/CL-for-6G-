import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.lines as mlines # Needed for custom legend
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

# --- Simulation Parameters ---
NUM_UES = 5
NUM_ANTENNAS = 64
NUM_BEAMS = NUM_ANTENNAS
ROAD_LENGTH = 500
BS_POSITION = ROAD_LENGTH / 2
FREQ = 28e9
TX_POWER_DBM = 30
NOISE_POWER_DBM = -70
NUM_TIMESTEPS = 4000
EVAL_TIMESTEPS = 400
TIMESTEP_DURATION = 0.01
BUFFER_SIZE = 10000
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
GAMMA = 0.99
SNR_THRESHOLD = 14.0
TARGET_UPDATE_FREQ = 50
PATH_LOSS_EXPONENT = 2.5
MAB_EXPLORATION_FACTOR = 2.0
VELOCITY_NORMALIZATION_FACTOR = 20.0
SNR_NORM_MIN = -10.0
SNR_NORM_MAX = 50.0
SMOOTHING_WINDOW = 50

# --- Time-Correlated Blockage Model Parameters ---
BLOCKAGE_ATTENUATION_DB = 25.0
P_BB = 0.85
P_UB = 0.03
# ---------------------------------------------

BANDWIDTH = 100e6

# --- Utility Function for Smoothing ---
def moving_average(data, window_size):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.size < window_size:
        return np.full(data.shape, np.nan)
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='same')
    half_window = window_size // 2
    smoothed[:half_window] = np.nan
    end_nan_idx = -(half_window -1) if window_size % 2 == 0 else -half_window
    if end_nan_idx < 0 and data.size > half_window:
       smoothed[end_nan_idx:] = np.nan
    return smoothed
# ---------------------------------------

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
BEAM_ANGLES = np.linspace(-np.pi/2, np.pi/2, NUM_BEAMS)

# Q-Network Definition (for Online Learning)
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
    # Restored indentation for forward method
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# --- Initialize models for OL (Online Learning) ---
input_size = 5
q_network = QNetwork(input_size, NUM_BEAMS).to(device)
target_network = QNetwork(input_size, NUM_BEAMS).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
replay_buffer = deque(maxlen=BUFFER_SIZE)

# MAB UCB1 Initialization
mab_counts = np.zeros((NUM_UES, NUM_BEAMS))
mab_values = np.zeros((NUM_UES, NUM_BEAMS))
mab_epsilon = 1e-6

# Channel configuration
try:
    channel_model = sn.channel.RayleighBlockFading(num_rx=NUM_UES, num_rx_ant=1, num_tx=1, num_tx_ant=NUM_ANTENNAS)
except AttributeError:
     print("Error: Could not find sn.channel.RayleighBlockFading."); exit()

# Path Loss Calculation
def compute_path_loss(distances):
    min_distance = 1.0; distances = np.maximum(distances, min_distance)
    path_loss_db = 32.45 + 10 * PATH_LOSS_EXPONENT * np.log10(distances + 1e-9)
    path_loss_linear = 10 ** (-path_loss_db / 10)
    return path_loss_linear, path_loss_db

# Compute SNR
def compute_snr(h_channel_all_ues, beam_idx, ue_idx, extra_attenuation_db=0.0):
    h_ue = h_channel_all_ues[ue_idx, :].astype(complex); beam = CODEBOOK[:, beam_idx].astype(complex)
    effective_signal_amplitude_sq = np.abs(np.dot(np.conj(h_ue), beam))**2
    noise_variance_relative = 10**((NOISE_POWER_DBM - TX_POWER_DBM) / 10)
    snr_linear = effective_signal_amplitude_sq / (noise_variance_relative + 1e-15)
    snr_db = 10 * np.log10(snr_linear + 1e-15) - extra_attenuation_db
    snr_db = np.clip(snr_db, -50, 50)
    return snr_db

# Channel Generation
def generate_channel(positions):
    try:
        h_tuple = channel_model(batch_size=1, num_time_steps=1)
        h_sionna = h_tuple[0] if isinstance(h_tuple, (list, tuple)) else h_tuple
        h = h_sionna.numpy().reshape(NUM_UES, NUM_ANTENNAS)
    except Exception as e:
        print(f"Error during Sionna channel model execution: {e}. Falling back to Gaussian.")
        h = (np.random.randn(NUM_UES, NUM_ANTENNAS) + 1j * np.random.randn(NUM_UES, NUM_ANTENNAS)) / np.sqrt(2)
    distances = np.abs(positions - BS_POSITION)
    path_loss_linear, _ = compute_path_loss(distances)
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

# --- UE Position Update ---
global_prev_positions = None
def update_positions(t):
    global global_prev_positions; positions = np.zeros(NUM_UES); velocities = np.zeros(NUM_UES)
    for i in range(NUM_UES):
        freq = 0.01 + i * 0.005; movement_range = 200
        positions[i] = BS_POSITION + movement_range * np.sin(freq * t * TIMESTEP_DURATION); positions[i] = np.clip(positions[i], 0, ROAD_LENGTH)
    if global_prev_positions is not None: velocities = (positions - global_prev_positions) / TIMESTEP_DURATION
    global_prev_positions = positions.copy()
    return positions, velocities

# --- Utility Functions ---
def compute_relative_angles(positions):
    y_distance = 10.0; x_distance = positions - BS_POSITION
    return np.arctan2(x_distance, y_distance)
def compute_distances(positions):
    distances_actual = np.abs(positions - BS_POSITION); normalized_distances = distances_actual / ROAD_LENGTH
    return normalized_distances, distances_actual
def normalize_snr(snr_db): return np.clip((snr_db - SNR_NORM_MIN) / (SNR_NORM_MAX - SNR_NORM_MIN), 0.0, 1.0)
def normalize_velocity(velocity_mps): return np.clip(velocity_mps / VELOCITY_NORMALIZATION_FACTOR, -1.0, 1.0)

# --- Reward and Metrics Computation ---
def compute_reward(throughput, snr, prev_snr, energy, accuracy_per_ue):
      w_tput=1.0; w_stab=0.2; w_energy=-0.3; w_acc=0.5
      throughput_reward=w_tput*(throughput/1500.0); stability_bonus=w_stab/(1+abs(snr-prev_snr)+1e-6)
      energy_penalty=w_energy*(max(0,energy)/10); accuracy_reward=w_acc*np.mean(accuracy_per_ue)
      return throughput_reward + stability_bonus + energy_penalty + accuracy_reward
def compute_throughput(snr_db):
    snr_linear=10**(snr_db/10); snr_linear=min(snr_linear, 1e15)
    throughput_bps=BANDWIDTH*np.log2(1+snr_linear); return throughput_bps/1e6
def compute_latency(avg_throughput_mbps):
    base_latency=0.5; max_additional_latency=5
    if not np.isfinite(avg_throughput_mbps): return base_latency+max_additional_latency
    exp_term=np.clip((avg_throughput_mbps-400)/100,-50,50); return base_latency+max_additional_latency/(1+np.exp(exp_term))
def compute_energy(snr_db,distance_actual):
    base_energy_mj=3.0; snr_factor=0.1*max(0,snr_db); distance_factor=0.01*distance_actual
    interaction_factor=0.05*max(0,snr_db)*(distance_actual/100); energy=base_energy_mj+snr_factor+distance_factor+interaction_factor
    return max(0.1,energy)

# OL Training Function
def train_q_network():
    if len(replay_buffer)<BATCH_SIZE: return 0.0
    batch_indices=np.random.choice(len(replay_buffer),BATCH_SIZE,replace=False); batch_data=[replay_buffer[i] for i in batch_indices]
    states,actions,rewards,next_states=zip(*batch_data); rewards_scalar=torch.tensor([r for r in rewards],dtype=torch.float32).to(device)
    states_np=np.array(states,dtype=np.float32); next_states_np=np.array(next_states,dtype=np.float32); actions_np=np.array(actions,dtype=np.int64)
    all_states_tensor=torch.from_numpy(states_np).to(device); all_actions_tensor=torch.from_numpy(actions_np).to(device); all_next_states_tensor=torch.from_numpy(next_states_np).to(device)
    batch_size_actual=all_states_tensor.shape[0]; num_ues_actual=all_states_tensor.shape[1]; state_dim=all_states_tensor.shape[2]
    if state_dim!=input_size: print(f"ERROR: State dimension mismatch! Expected {input_size}, Got {state_dim}"); return -1.0
    reshaped_states=all_states_tensor.view(batch_size_actual*num_ues_actual,state_dim); reshaped_next_states=all_next_states_tensor.view(batch_size_actual*num_ues_actual,state_dim)
    q_values_all=q_network(reshaped_states); flat_actions=all_actions_tensor.view(-1); current_q_values=q_values_all.gather(1,flat_actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_q_values_online_all=q_network(reshaped_next_states); best_next_actions=next_q_values_online_all.argmax(1)
        next_q_values_target_all=target_network(reshaped_next_states); max_next_q_values=next_q_values_target_all.gather(1,best_next_actions.unsqueeze(-1)).squeeze(-1)
    expanded_rewards=rewards_scalar.unsqueeze(1).expand(-1,num_ues_actual).reshape(-1); expected_q_values=expanded_rewards+GAMMA*max_next_q_values
    loss=nn.MSELoss()(current_q_values,expected_q_values); total_batch_loss=loss.item()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return total_batch_loss

# Baseline Metric Calculation Functions
def angle_heuristic_beam_switching(snr_heuristic,positions_actual):
    _,distances_actual=compute_distances(positions_actual); throughputs=np.array([compute_throughput(snr) for snr in snr_heuristic])
    avg_throughput=np.mean(throughputs); latency=compute_latency(avg_throughput)
    energies=np.array([compute_energy(snr_heuristic[i],distances_actual[i]) for i in range(NUM_UES)]); avg_energy=np.mean(energies)
    accuracies=(snr_heuristic>SNR_THRESHOLD).astype(float); avg_accuracy=np.mean(accuracies); avg_snr=np.mean(snr_heuristic)
    return latency,avg_throughput,avg_energy,avg_accuracy,avg_snr
def mab_ucb1_action(ue_idx,t,mab_counts,mab_values,exploration_factor=2.0):
    unexplored_arms=np.where(mab_counts[ue_idx,:]==0)[0]
    if len(unexplored_arms)>0: return unexplored_arms[0]
    total_counts_ue=np.sum(mab_counts[ue_idx,:]); ucb_values=np.zeros(NUM_BEAMS)
    for beam_idx in range(NUM_BEAMS):
        count=mab_counts[ue_idx,beam_idx]
        if count==0: mean_reward=float('inf'); exploration_bonus=float('inf')
        else: mean_reward=mab_values[ue_idx,beam_idx]/count; exploration_bonus=np.sqrt(exploration_factor*np.log(max(1,total_counts_ue))/(count+mab_epsilon))
        ucb_values[beam_idx]=mean_reward+exploration_bonus
    return np.argmax(ucb_values)
def mab_ucb1_beam_switching(snr_mab,positions_actual):
     _,distances_actual=compute_distances(positions_actual); throughputs=np.array([compute_throughput(snr) for snr in snr_mab])
     avg_throughput=np.mean(throughputs); latency=compute_latency(avg_throughput)
     energies=np.array([compute_energy(snr_mab[i],distances_actual[i]) for i in range(NUM_UES)]); avg_energy=np.mean(energies)
     accuracies=(snr_mab>SNR_THRESHOLD).astype(float); avg_accuracy=np.mean(accuracies); avg_snr=np.mean(snr_mab)
     return latency,avg_throughput,avg_energy,avg_accuracy,avg_snr

# --- Training Phase ---
print("--- Starting Training Phase ---")
results_ol = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
results_angle_heuristic = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
results_mab_ucb = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
snr_log_ol = []; snr_log_angle_heuristic = []; snr_log_mab_ucb = []
epsilon = 1.0
prev_snr_ol = np.zeros(NUM_UES)
prev_is_blocked = np.zeros(NUM_UES, dtype=bool)
initial_positions, _ = update_positions(0)
initial_h_channel = generate_channel(initial_positions)
prev_snr_ol = initial_beam_scan(initial_h_channel)
print(f"Initial Avg SNR from Scan: {np.mean(prev_snr_ol):.2f} dB")

for t in range(NUM_TIMESTEPS):
    positions, velocities = update_positions(t)
    angles = compute_relative_angles(positions)
    norm_distances, actual_distances = compute_distances(positions)
    h_channel = generate_channel(positions)
    norm_velocities = normalize_velocity(velocities)

    heuristic_beam_indices = np.zeros(NUM_UES, dtype=int)
    current_is_blocked = np.zeros(NUM_UES, dtype=bool)
    # Restored indentation for this loop
    for i in range(NUM_UES):
        heuristic_beam_indices[i] = np.argmin(np.abs(BEAM_ANGLES - angles[i]))
        rand_val = np.random.rand()
        # Restored indentation for if/else
        if prev_is_blocked[i]:
            current_is_blocked[i] = (rand_val < P_BB)
        else:
            current_is_blocked[i] = (rand_val < P_UB)

    snr_ol = np.zeros(NUM_UES); snr_heuristic = np.zeros(NUM_UES); snr_mab = np.zeros(NUM_UES)
    actions_ol = np.zeros(NUM_UES, dtype=int); actions_mab = np.zeros(NUM_UES, dtype=int)
    norm_prev_snr_ol = normalize_snr(prev_snr_ol); prev_blocked_float = prev_is_blocked.astype(float)
    state_ol = np.array([[ angles[i], norm_prev_snr_ol[i], norm_distances[i], norm_velocities[i], prev_blocked_float[i] ] for i in range(NUM_UES)])

    # Restored indentation for the main loop logic
    for i in range(NUM_UES):
        action_h = heuristic_beam_indices[i]
        action_mab = mab_ucb1_action(i, t, mab_counts, mab_values, MAB_EXPLORATION_FACTOR)
        actions_mab[i] = action_mab

        # Restored indentation for if/else and 'with' block
        if np.random.rand() < epsilon:
            action_ol = np.random.randint(NUM_BEAMS)
        else:
            with torch.no_grad():
                state_tensor_i = torch.tensor(state_ol[i], dtype=torch.float32).unsqueeze(0).to(device)
                action_ol = q_network(state_tensor_i).argmax().item()
        actions_ol[i] = action_ol

        attenuation_h = BLOCKAGE_ATTENUATION_DB if action_h == heuristic_beam_indices[i] and current_is_blocked[i] else 0.0
        attenuation_mab = BLOCKAGE_ATTENUATION_DB if action_mab == heuristic_beam_indices[i] and current_is_blocked[i] else 0.0
        attenuation_ol = BLOCKAGE_ATTENUATION_DB if action_ol == heuristic_beam_indices[i] and current_is_blocked[i] else 0.0

        snr_heuristic[i] = compute_snr(h_channel, action_h, i, extra_attenuation_db=attenuation_h)
        snr_mab[i] = compute_snr(h_channel, action_mab, i, extra_attenuation_db=attenuation_mab)
        snr_ol[i] = compute_snr(h_channel, action_ol, i, extra_attenuation_db=attenuation_ol)

        reward_mab = snr_mab[i]; mab_counts[i, action_mab] += 1; mab_values[i, action_mab] += reward_mab

    latency_h, throughput_h, energy_h, accuracy_h, avg_snr_h = angle_heuristic_beam_switching(snr_heuristic, positions)
    results_angle_heuristic["latency"].append(latency_h); results_angle_heuristic["throughput"].append(throughput_h); results_angle_heuristic["energy"].append(energy_h); results_angle_heuristic["accuracy"].append(accuracy_h); snr_log_angle_heuristic.append(avg_snr_h)
    latency_mab, throughput_mab, energy_mab, accuracy_mab, avg_snr_mab = mab_ucb1_beam_switching(snr_mab, positions)
    results_mab_ucb["latency"].append(latency_mab); results_mab_ucb["throughput"].append(throughput_mab); results_mab_ucb["energy"].append(energy_mab); results_mab_ucb["accuracy"].append(accuracy_mab); snr_log_mab_ucb.append(avg_snr_mab)
    throughputs_ol = np.array([compute_throughput(snr) for snr in snr_ol]); avg_throughput_ol = np.mean(throughputs_ol); energies_ol = np.array([compute_energy(snr_ol[i], actual_distances[i]) for i in range(NUM_UES)]); avg_energy_ol = np.mean(energies_ol)
    avg_snr_ol = np.mean(snr_ol); latency_ol = compute_latency(avg_throughput_ol); accuracy_per_ue_ol = (snr_ol > SNR_THRESHOLD).astype(float); avg_accuracy_ol = np.mean(accuracy_per_ue_ol)
    reward_ol = compute_reward(avg_throughput_ol, avg_snr_ol, np.mean(prev_snr_ol), avg_energy_ol, accuracy_per_ue_ol)
    results_ol["latency"].append(latency_ol); results_ol["throughput"].append(avg_throughput_ol); results_ol["energy"].append(avg_energy_ol); results_ol["accuracy"].append(avg_accuracy_ol); snr_log_ol.append(avg_snr_ol)
    next_positions, next_velocities = update_positions(t + 1); global_prev_positions = positions.copy(); next_angles = compute_relative_angles(next_positions); next_norm_distances, _ = compute_distances(next_positions); next_norm_velocities = normalize_velocity(next_velocities)
    norm_current_snr_ol = normalize_snr(snr_ol); current_blocked_float = current_is_blocked.astype(float)
    next_state_ol = np.array([[ next_angles[i], norm_current_snr_ol[i], next_norm_distances[i], next_norm_velocities[i], current_blocked_float[i] ] for i in range(NUM_UES)])
    replay_buffer.append((state_ol, actions_ol, reward_ol, next_state_ol)); prev_snr_ol = snr_ol.copy(); prev_is_blocked = current_is_blocked.copy()
    loss_val = train_q_network();
    if (t + 1) % TARGET_UPDATE_FREQ == 0: target_network.load_state_dict(q_network.state_dict())
    epsilon = max(0.1, epsilon * 0.999);
    if (t + 1) % 200 == 0: print(f"Training Timestep: {t+1}/{NUM_TIMESTEPS}, Epsilon: {epsilon:.3f}, AvgLoss: {loss_val:.4f}, AvgSNR OL: {avg_snr_ol:.2f} dB")
print("--- Training Phase Completed ---")

# --- Evaluation Phase ---
print("--- Starting Evaluation Phase ---")
results_ol_eval = {"latency": [], "throughput": [], "energy": [], "accuracy": []}; results_angle_heuristic_eval = {"latency": [], "throughput": [], "energy": [], "accuracy": []}; results_mab_ucb_eval = {"latency": [], "throughput": [], "energy": [], "accuracy": []}
snr_log_ol_eval = []; snr_log_angle_heuristic_eval = []; snr_log_mab_ucb_eval = []
prev_snr_ol_eval = prev_snr_ol.copy(); prev_is_blocked_eval = np.zeros(NUM_UES, dtype=bool)
for t_eval in range(EVAL_TIMESTEPS):
    t = NUM_TIMESTEPS + t_eval; positions, velocities = update_positions(t); angles = compute_relative_angles(positions); norm_distances, actual_distances = compute_distances(positions); h_channel = generate_channel(positions); norm_velocities = normalize_velocity(velocities)
    heuristic_beam_indices = np.zeros(NUM_UES, dtype=int); current_is_blocked_eval = np.zeros(NUM_UES, dtype=bool)
    # Restored indentation for this loop
    for i in range(NUM_UES):
        heuristic_beam_indices[i] = np.argmin(np.abs(BEAM_ANGLES - angles[i]))
        rand_val = np.random.rand()
        # Restored indentation for if/else
        if prev_is_blocked_eval[i]:
            current_is_blocked_eval[i] = (rand_val < P_BB)
        else:
            current_is_blocked_eval[i] = (rand_val < P_UB)

    snr_ol = np.zeros(NUM_UES); snr_heuristic = np.zeros(NUM_UES); snr_mab = np.zeros(NUM_UES); actions_ol = np.zeros(NUM_UES, dtype=int); actions_mab = np.zeros(NUM_UES, dtype=int)
    norm_prev_snr_ol_eval = normalize_snr(prev_snr_ol_eval); prev_blocked_eval_float = prev_is_blocked_eval.astype(float)
    state_ol = np.array([[ angles[i], norm_prev_snr_ol_eval[i], norm_distances[i], norm_velocities[i], prev_blocked_eval_float[i] ] for i in range(NUM_UES)])

    # Restored indentation for the main evaluation loop logic
    for i in range(NUM_UES):
        action_h = heuristic_beam_indices[i]
        counts_i = mab_counts[i, :]
        # Restored indentation for if/else
        if np.all(counts_i == 0):
            action_mab = np.random.randint(NUM_BEAMS)
        else:
             mean_rewards = np.full(NUM_BEAMS, -np.inf); valid_indices = counts_i > 0
             mean_rewards[valid_indices] = np.divide(mab_values[i, valid_indices], counts_i[valid_indices]);
             action_mab = np.argmax(mean_rewards)
        actions_mab[i] = action_mab

        # Restored indentation for 'with' block
        with torch.no_grad():
           state_tensor_i = torch.tensor(state_ol[i], dtype=torch.float32).unsqueeze(0).to(device)
           action_ol = q_network(state_tensor_i).argmax().item()
        actions_ol[i] = action_ol

        attenuation_h = BLOCKAGE_ATTENUATION_DB if action_h == heuristic_beam_indices[i] and current_is_blocked_eval[i] else 0.0
        attenuation_mab = BLOCKAGE_ATTENUATION_DB if action_mab == heuristic_beam_indices[i] and current_is_blocked_eval[i] else 0.0
        attenuation_ol = BLOCKAGE_ATTENUATION_DB if action_ol == heuristic_beam_indices[i] and current_is_blocked_eval[i] else 0.0

        snr_heuristic[i] = compute_snr(h_channel, action_h, i, extra_attenuation_db=attenuation_h)
        snr_mab[i] = compute_snr(h_channel, action_mab, i, extra_attenuation_db=attenuation_mab)
        snr_ol[i] = compute_snr(h_channel, action_ol, i, extra_attenuation_db=attenuation_ol)

    latency_h, throughput_h, energy_h, accuracy_h, avg_snr_h = angle_heuristic_beam_switching(snr_heuristic, positions)
    results_angle_heuristic_eval["latency"].append(latency_h); results_angle_heuristic_eval["throughput"].append(throughput_h); results_angle_heuristic_eval["energy"].append(energy_h); results_angle_heuristic_eval["accuracy"].append(accuracy_h); snr_log_angle_heuristic_eval.append(avg_snr_h)
    latency_mab, throughput_mab, energy_mab, accuracy_mab, avg_snr_mab = mab_ucb1_beam_switching(snr_mab, positions)
    results_mab_ucb_eval["latency"].append(latency_mab); results_mab_ucb_eval["throughput"].append(throughput_mab); results_mab_ucb_eval["energy"].append(energy_mab); results_mab_ucb_eval["accuracy"].append(accuracy_mab); snr_log_mab_ucb_eval.append(avg_snr_mab)
    throughputs_ol = np.array([compute_throughput(snr) for snr in snr_ol]); avg_throughput_ol = np.mean(throughputs_ol); energies_ol = np.array([compute_energy(snr_ol[i], actual_distances[i]) for i in range(NUM_UES)]); avg_energy_ol = np.mean(energies_ol)
    avg_snr_ol = np.mean(snr_ol); latency_ol = compute_latency(avg_throughput_ol); accuracy_ol = np.mean((snr_ol > SNR_THRESHOLD).astype(float))
    results_ol_eval["latency"].append(latency_ol); results_ol_eval["throughput"].append(avg_throughput_ol); results_ol_eval["energy"].append(avg_energy_ol); results_ol_eval["accuracy"].append(accuracy_ol); snr_log_ol_eval.append(avg_snr_ol)
    prev_snr_ol_eval = snr_ol.copy(); prev_is_blocked_eval = current_is_blocked_eval.copy();
    if (t_eval + 1) % 100 == 0: print(f"Evaluation Timestep: {t_eval+1}/{EVAL_TIMESTEPS}, AvgSNR OL: {avg_snr_ol:.2f} dB")
print("--- Evaluation Phase Completed ---")

# --- Plot Results ---
print("--- Plotting Results ---")
plt.figure(figsize=(15, 10))
metrics = ["latency", "throughput", "energy", "accuracy"]
ylabels = ["Latency (ms)", "Throughput (Mbps)", "Energy (mJ)", "Accuracy"]
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
title_fontsize = 12; title_fontweight = 'bold'; ylabel_fontweight = 'bold'; xlabel_fontweight = 'bold'; xlabel_fontsize = 10
legend_fontsize = 'large'; legend_title_fontsize = 'large'
color_ol_eval = "green"; color_heuristic = "orange"; color_mab = "red"
eval_plot_range = range(EVAL_TIMESTEPS)

for i, (key, ylabel) in enumerate(zip(metrics, ylabels), 1):
    plt.subplot(3, 2, i); plot_label_prefix = f"{subplot_labels[i-1]} "
    data_ol_eval = results_ol_eval[key]; data_heuristic_eval = results_angle_heuristic_eval[key]; data_mab_eval = results_mab_ucb_eval[key]
    smoothed_ol_eval = moving_average(data_ol_eval, SMOOTHING_WINDOW); smoothed_heuristic_eval = moving_average(data_heuristic_eval, SMOOTHING_WINDOW); smoothed_mab_eval = moving_average(data_mab_eval, SMOOTHING_WINDOW)
    plot_title = f"{plot_label_prefix}Smoothed {key.capitalize()} Comparison (Evaluation)"
    plt.plot(eval_plot_range, smoothed_ol_eval, color=color_ol_eval, linewidth=1.5)
    plt.plot(eval_plot_range, smoothed_heuristic_eval, color=color_heuristic, linestyle="--", linewidth=1.0)
    plt.plot(eval_plot_range, smoothed_mab_eval, color=color_mab, linestyle="-.", linewidth=1.0)
    plt.xlabel("Evaluation Timestep", fontweight=xlabel_fontweight, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontweight=ylabel_fontweight)
    plt.title(plot_title, fontweight=title_fontweight, fontsize=title_fontsize)
    plt.grid(True, linestyle=':', alpha=0.6); plt.xlim(0, EVAL_TIMESTEPS)

# Plot Smoothed SNR (Subplot 5)
plt.subplot(3, 2, 5); plot_label_prefix_snr = f"{subplot_labels[4]} "
snr_ol_eval_smoothed = moving_average(snr_log_ol_eval, SMOOTHING_WINDOW); snr_heuristic_eval_smoothed = moving_average(snr_log_angle_heuristic_eval, SMOOTHING_WINDOW); snr_mab_eval_smoothed = moving_average(snr_log_mab_ucb_eval, SMOOTHING_WINDOW)
plt.plot(eval_plot_range, snr_ol_eval_smoothed, color=color_ol_eval, linewidth=1.5)
plt.plot(eval_plot_range, snr_heuristic_eval_smoothed, color=color_heuristic, linestyle="--", linewidth=1.0)
plt.plot(eval_plot_range, snr_mab_eval_smoothed, color=color_mab, linestyle="-.", linewidth=1.0)
plt.xlabel("Evaluation Timestep", fontweight=xlabel_fontweight, fontsize=xlabel_fontsize)
plt.ylabel("Average SNR (dB)", fontweight=ylabel_fontweight)
plt.title(f"{plot_label_prefix_snr}Smoothed Average SNR Comparison (Evaluation)", fontweight=title_fontweight, fontsize=title_fontsize)
plt.grid(True, linestyle=':', alpha=0.6); plt.xlim(0, EVAL_TIMESTEPS)

# --- Create Legend in Empty Subplot (Position 3,2 -> Index 6) ---
ax_legend = plt.subplot(3, 2, 6)
line_ol_eval = mlines.Line2D([], [], color=color_ol_eval, linestyle='-', linewidth=1.5, label='OL (Eval)')
line_heuristic_eval = mlines.Line2D([], [], color=color_heuristic, linestyle='--', linewidth=1.0, label='Heuristic (Eval)')
line_mab_eval = mlines.Line2D([], [], color=color_mab, linestyle='-.', linewidth=1.0, label='MAB UCB1 (Eval)')
ax_legend.legend(handles=[line_ol_eval, line_heuristic_eval, line_mab_eval], loc='center',
                 fontsize=legend_fontsize, frameon=True, title="Methods (Smoothed Evaluation)",
                 title_fontsize=legend_title_fontsize)
ax_legend.axis('off')

# --- Final Figure Adjustments ---
plt.tight_layout(rect=[0, 0.03, 1, 0.98])

# Save the figure
try:
    plt.savefig("results_final_evaluation_smoothed.png", dpi=300, bbox_inches='tight')
    print("Results plot saved to results_final_evaluation_smoothed.png")
except Exception as e:
    print(f"Error saving plot: {e}")

# plt.show()

# --- Print average metrics ---
print("\n--- Average Metrics (Evaluation Phase Only) ---")
def print_avg_results(label, results_dict, snr_log):
    latency_list = results_dict.get('latency', []); throughput_list = results_dict.get('throughput', []); energy_list = results_dict.get('energy', []); accuracy_list = results_dict.get('accuracy', []); snr_list = snr_log
    if not latency_list or not snr_list: print(f"{label}: No evaluation data to report."); return
    print(f"{label}:")
    print(f"  Raw Avg Latency:    {np.mean(latency_list):.2f} ms")
    print(f"  Raw Avg Throughput: {np.mean(throughput_list):.2f} Mbps")
    print(f"  Raw Avg Energy:     {np.mean(energy_list):.2f} mJ")
    print(f"  Raw Avg Accuracy:   {np.mean(accuracy_list):.2f}")
    print(f"  Raw Avg SNR:        {np.mean(snr_list):.2f} dB")

print_avg_results("OL Results (Evaluation)", results_ol_eval, snr_log_ol_eval)
print_avg_results("Angle Heuristic Results (Evaluation)", results_angle_heuristic_eval, snr_log_angle_heuristic_eval)
print_avg_results("MAB UCB1 Results (Evaluation)", results_mab_ucb_eval, snr_log_mab_ucb_eval)
print("\n--- Script Finished ---")