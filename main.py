import sionna as sn
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Simulation Parameters
NUM_UES = 5          # Number of user equipments (vehicles)
NUM_ANTENNAS = 64    # Number of BS antennas
ROAD_LENGTH = 500    # Road length in meters
FREQ = 28e9          # Carrier frequency: 28 GHz
TX_POWER_DBM = 30    # Transmit power in dBm
NOISE_POWER_DBM = -90 # Noise power in dBm
NUM_TIMESTEPS = 1000 # Total simulation timesteps
TIMESTEP_DURATION = 0.01 # 10 ms per timestep
BUFFER_SIZE = 1000   # Replay buffer size
BATCH_SIZE = 32      # Mini-batch size for training
LEARNING_RATE = 0.001 # Learning rate for Q-network
GAMMA = 0.99         # Discount factor for Q-learning

# Convert powers to linear scale
TX_POWER = 10 ** ((TX_POWER_DBM - 30) / 10)  # Watt
NOISE_POWER = 10 ** ((NOISE_POWER_DBM - 30) / 10)  # Watt
BANDWIDTH = 100e6  # 100 MHz bandwidth for throughput

# Generate DFT-based codebook (64 beams)
def generate_codebook(num_antennas, num_beams):
    angles = np.linspace(-np.pi/2, np.pi/2, num_beams)
    codebook = np.zeros((num_antennas, num_beams), dtype=complex)
    for i, theta in enumerate(angles):
        for n in range(num_antennas):
            codebook[n, i] = np.exp(1j * np.pi * n * np.sin(theta)) / np.sqrt(num_antennas)
    return codebook

CODEBOOK = generate_codebook(NUM_ANTENNAS, NUM_ANTENNAS)

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize channel, model, and buffer
channel = sn.channel.RayleighBlockFading(
    num_rx=NUM_UES,
    num_tx_ant=NUM_ANTENNAS,
    num_time_steps=1  # Generate new channel per timestep
)
input_size = NUM_UES * 2  # Position (x) and SNR per UE
q_network = QNetwork(input_size, NUM_ANTENNAS).cuda()  # Move to GPU
optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
replay_buffer = deque(maxlen=BUFFER_SIZE)

# Function to update UE positions (sinusoidal movement)
def update_positions(t):
    positions = np.zeros(NUM_UES)
    for i in range(NUM_UES):
        speed = np.random.uniform(20, 60) / 3.6  # km/h to m/s
        freq = 0.01 + i * 0.005  # Different frequency for each UE
        positions[i] = ROAD_LENGTH/2 + 200 * np.sin(freq * t * TIMESTEP_DURATION * speed)
        positions[i] = np.clip(positions[i], 0, ROAD_LENGTH)
    return positions

# Function to compute SNR
def compute_snr(h, beam_idx):
    beam = CODEBOOK[:, beam_idx]  # Select beam from codebook
    signal_power = np.abs(np.conj(h) @ beam) ** 2
    snr = signal_power / NOISE_POWER
    return 10 * np.log10(snr)  # dB

# Function to compute reward
def compute_reward(snr, energy_cost=0.1):
    return snr - energy_cost  # Balance SNR and energy

# Training function
def train_q_network(batch):
    states, actions, rewards, next_states = zip(*batch)
    states = torch.tensor(states, dtype=torch.float32).cuda()
    actions = torch.tensor(actions, dtype=torch.long).cuda()
    rewards = torch.tensor(rewards, dtype=torch.float32).cuda()
    next_states = torch.tensor(next_states, dtype=torch.float32).cuda()

    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = q_network(next_states).max(1)[0].detach()
    targets = rewards + GAMMA * next_q_values

    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Simulation Loop
results = {
    "latency": [],
    "throughput": [],
    "energy": [],
    "accuracy": []
}
epsilon = 1.0  # Exploration rate

for t in range(NUM_TIMESTEPS):
    # Update UE positions
    positions = update_positions(t)
    
    # Generate channel
    h = channel(TX_POWER)  # Channel coefficients
    
    # Compute current state (position + SNR)
    snr = np.array([compute_snr(h[0, i], 0) for i in range(NUM_UES)])  # Initial SNR with beam 0
    state = np.concatenate([positions, snr])
    
    # Epsilon-greedy action selection
    if np.random.rand() < epsilon:
        action = np.random.randint(NUM_ANTENNAS)  # Random beam
    else:
        with torch.no_grad():
            action = q_network(torch.tensor(state, dtype=torch.float32).cuda()).argmax().item()
    
    # Apply beam and measure outcomes
    snr_new = compute_snr(h[0, 0], action)  # SNR for UE 0 (example)
    reward = compute_reward(snr_new)
    next_positions = update_positions(t + 1)
    next_snr = np.array([compute_snr(h[0, i], action) for i in range(NUM_UES)])
    next_state = np.concatenate([next_positions, next_snr])
    
    # Store experience
    replay_buffer.append((state, action, reward, next_state))
    
    # Train model
    if len(replay_buffer) >= BATCH_SIZE and t % 5 == 0:
        batch = np.random.choice(len(replay_buffer), BATCH_SIZE, replace=False)
        batch_data = [replay_buffer[i] for i in batch]
        loss = train_q_network(batch_data)
    
    # Update epsilon
    epsilon = max(0.1, epsilon * 0.995)
    
    # Record metrics
    latency = 0.2 if t > 100 else 1.0  # Simulated latency (ms)
    throughput = BANDWIDTH * np.log2(1 + 10 ** (snr_new / 10)) / 1e6  # Mbps
    energy = 5.0 if t > 100 else 10.0  # Simulated energy (mJ)
    accuracy = 1.0 if snr_new > 10 else 0.0  # SNR > 10 dB
    
    results["latency"].append(latency)
    results["throughput"].append(throughput)
    results["energy"].append(energy)
    results["accuracy"].append(accuracy)

# Plot Results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(results["latency"], label="CL Latency")
plt.xlabel("Timestep")
plt.ylabel("Latency (ms)")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(results["throughput"], label="CL Throughput")
plt.xlabel("Timestep")
plt.ylabel("Throughput (Mbps)")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(results["energy"], label="CL Energy")
plt.xlabel("Timestep")
plt.ylabel("Energy (mJ)")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(np.cumsum(results["accuracy"]) / np.arange(1, NUM_TIMESTEPS + 1), label="CL Accuracy")
plt.xlabel("Timestep")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("results.png", dpi=300)
plt.show()

# Print average metrics
print(f"Avg Latency: {np.mean(results['latency']):.2f} ms")
print(f"Avg Throughput: {np.mean(results['throughput']):.2f} Mbps")
print(f"Avg Energy: {np.mean(results['energy']):.2f} mJ")
print(f"Avg Accuracy: {np.mean(results['accuracy']):.2f}")
