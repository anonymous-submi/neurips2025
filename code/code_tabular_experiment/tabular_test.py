import numpy as np
import matplotlib.pyplot as plt

### experiment for tabular setting

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14
})

N0 = 200                  # Number of good states
N1 = 1000                # Number of recover states
A = 100                   # Action space size
horizon = 100              # Episode length
gamma = 8/horizon               # Slip probability

batch_size = 200          # Parallel agents
total_rounds = 220        # Training rounds
switch_points = [2, 8, 32] # Rounds to Switch from BC to DAgger 


# Expert configuration
np.random.seed(1)
BAD_RECOVERABLE = N0 + N1
BAD_ABSORBING = N0 + N1 + 1
expert_action_map = np.random.randint(0, A, N0 + N1 + 2)


class BatchPolicy:
    def __init__(self, batch_size, num_states, num_actions):
        self.batch_size = batch_size
        self.num_states = num_states
        self.action_table = np.random.randint(0, num_actions, (batch_size, num_states))
        self.annotation_mask = np.zeros((batch_size, num_states), dtype=bool)

    def update(self, states, expert_action_map):
        """
        states: shape [batch_size, num_samples]
        expert_action_map: shape [batch_size, num_samples]
        """
        for b in range(self.batch_size):
            # Get unique states and their first occurrence indices
            unique_states, first_occurrence = np.unique(states[b], return_index=True)
            
            # Get corresponding expert actions for first occurrences
            selected_actions = expert_action_map[b, first_occurrence]
            
            # Find which states haven't been annotated yet
            update_mask = ~self.annotation_mask[b, unique_states]
            
            # Update action table and annotation mask
            self.action_table[b, unique_states[update_mask]] = selected_actions[update_mask]
            self.annotation_mask[b, unique_states[update_mask]] = True

    def act(self, states):
        """states: shape [batch_size]"""
        return self.action_table[np.arange(self.batch_size), states]


def vectorized_transition(states, actions):
    next_states = np.zeros_like(states)
    valid = (actions == expert_action_map[states])
    
    # 0-error states
    mask_0 = states < N0
    if mask_0.any():
        slip = np.random.rand(mask_0.sum()) < gamma
        next_states[mask_0] = np.where(
            valid[mask_0],
            np.where(
                ~slip,
                np.random.randint(0, N0, mask_0.sum()),  # Stay in N0
                np.random.randint(N0, N0+N1, mask_0.sum())  # Slip to N1
            ),
            BAD_ABSORBING  # Wrong action → absorbing bad
        )
    
    # 1-error states
    mask_1 = (states >= N0) & (states < N0+N1)
    if mask_1.any():
        next_states[mask_1] = np.where(
            valid[mask_1],
            np.random.randint(0, N0, mask_1.sum()),  # Recover to N0
            N0 + N1  # recoverable Bad state
        )
    
    # Recoverable bad state
    mask_rec = (states == BAD_RECOVERABLE)
    if mask_rec.any():
        next_states[mask_rec] = np.where(
            valid[mask_rec],
            np.random.randint(N0, N0+N1, mask_rec.sum()),  # Correct → back to N1
            BAD_RECOVERABLE  # Wrong → recoverable bad
        )
    
    # Absorbing bad state
    mask_abs = (states == BAD_ABSORBING)
    next_states[mask_abs] = BAD_ABSORBING
    
    return next_states


def init_policies():
    return {
        'BC': BatchPolicy(batch_size, N0 + N1 + 2, A),
        'DAgger': BatchPolicy(batch_size, N0 + N1 + 2, A),
        **{f'Hybrid_{sp}': BatchPolicy(batch_size, N0 + N1 + 2, A) 
           for sp in switch_points}
    }

def collect_expert_data():
    """Generate expert trajectories with environment dynamics (including slips)"""
    states = np.zeros((batch_size, horizon), dtype=int)
    actions = np.zeros_like(states)
    
    # Start in 0-error states
    current_states = np.random.randint(0, N0, batch_size)
    
    for t in range(horizon):
        # Expert takes optimal actions
        current_actions = expert_action_map[current_states]
        states[:, t] = current_states
        actions[:, t] = current_actions
        
        # Transition with environment dynamics
        current_states = vectorized_transition(current_states, current_actions)
    
    return states, actions


def estimate_expert_cost(num_episodes=100):
    """Return expert's per-agent cost and failure arrays over all episodes."""
    all_costs = []
    all_failures = []

    for _ in range(num_episodes):
        states, actions = collect_expert_data()  # shape: [batch_size, horizon]

        # Per-agent cost: proportion of timesteps in bad states (>= N0)
        cost_per_agent = np.sum(states >= N0, axis=1)   # shape: [batch_size]

        # Per-agent failure: number of visits to absorbing bad state (= N0 + N1)
        fail_per_agent = np.sum(states == (N0 + N1), axis=1)  # shape: [batch_size]

        all_costs.append(cost_per_agent)
        all_failures.append(fail_per_agent)

    # Stack and average across episodes
    all_costs = np.stack(all_costs)     # shape: [num_episodes, batch_size]
    all_failures = np.stack(all_failures)

    # Return as list of per-agent values (averaged over episodes)
    avg_cost_per_agent = np.mean(all_costs, axis=0).tolist()     # shape: [batch_size]
    avg_fail_per_agent = np.mean(all_failures, axis=0).tolist()

    return avg_cost_per_agent, avg_fail_per_agent


### main function

def train(policies):
    metrics = {name: {'cost': [], 'fail': [], 'n0_coverage': [], 'n1_coverage': []} 
               for name in policies}
    
    
    for round_idx in range(total_rounds):
        print(f"Training Round {round_idx+1}/{total_rounds}")

        # Collect  expert trajectories (states and actions)
        expert_traj_states, expert_traj_actions = collect_expert_data()
        
        # Update BC policy with trajectory actions
        policies['BC'].update(expert_traj_states, expert_traj_actions)
        
        # DAgger data collection
        dagger_states = np.zeros((batch_size, horizon), dtype=int)
        dagger_actions = np.zeros_like(dagger_states)
        current_states = np.random.randint(0, N0, batch_size)
        
        for t in range(horizon):
            actions = policies['DAgger'].act(current_states)
            dagger_states[:, t] = current_states
            # Use global expert action map for corrections
            dagger_actions[:, t] = expert_action_map[current_states]
            current_states = vectorized_transition(current_states, actions)
        
        policies['DAgger'].update(dagger_states, dagger_actions)
        
        # Hybrid policies update
        for sp in switch_points:
            policy_name = f'Hybrid_{sp}'
            if round_idx < sp:
                # BC phase uses trajectory data
                policies[policy_name].update(expert_traj_states, expert_traj_actions)
            else:
                # DAgger phase uses global expert map
                hybrid_states = np.zeros((batch_size, horizon), dtype=int)
                hybrid_actions = np.zeros_like(hybrid_states)
                current_states = np.random.randint(0, N0, batch_size)
                # n0_mistake_flags[:] = False
                
                for t in range(horizon):
                    actions = policies[policy_name].act(current_states)
                    hybrid_states[:, t] = current_states
                    hybrid_actions[:, t] = expert_action_map[current_states]
                    current_states = vectorized_transition(current_states, actions)
                
                policies[policy_name].update(hybrid_states, hybrid_actions)
        
        # Evaluate all policies
        if (round_idx + 1) % 1 == 0:
            test_states = np.random.randint(0, N0, (batch_size, horizon))
            for name, policy in policies.items():
                cost_per_agent = np.zeros(batch_size)
                fail_per_agent = np.zeros(batch_size)
                current_states = test_states[:, 0].copy()
                for t in range(horizon):
                    actions = policy.act(current_states)
                    cost_per_agent += (current_states >= N0)
                    next_states = vectorized_transition(current_states, actions)
                    fail_per_agent += (next_states >= N0 + N1)
                    current_states = next_states.copy()

                agent_costs = cost_per_agent 
                agent_fails = fail_per_agent

                # Store raw per-agent vectors
                metrics[name]['cost'].append(agent_costs.tolist())      # shape: [batch_size]
                metrics[name]['fail'].append(agent_fails.tolist())

                # Calculate annotation-based coverage
                n0_covered = np.sum(policy.annotation_mask[:, :N0], axis=1)  # [batch_size]
                n1_covered = np.sum(policy.annotation_mask[:, N0:N0+N1], axis=1)


                metrics[name]['n0_coverage'].append((n0_covered / N0).tolist())  # list of shape [batch_size]
                metrics[name]['n1_coverage'].append((n1_covered / N1).tolist())

    return metrics



### Execution & Visualization

policies = init_policies()
metrics = train(policies)

print('training done')

expert_cost_per_agent, expert_fail_per_agent = estimate_expert_cost()

dummy_coverage = [1.0] * batch_size  # Expert has no annotation mask

metrics['expert'] = {
    'cost': [expert_cost_per_agent] * total_rounds,
    'fail': [expert_fail_per_agent] * total_rounds,
    'n0_coverage': [dummy_coverage] * total_rounds,
    'n1_coverage': [dummy_coverage] * total_rounds,
}

print('start ploting')

# Mapping
name_mapping = {
    'expert': 'Expert',
    'BC': 'BC',
    'DAgger': 'DAgger',
    'Hybrid_2': 'WSD (200)',
    'Hybrid_8': 'WSD (800)',
    'Hybrid_32': 'WSD (3200)',
}
color_mapping = {
    'expert': 'C0',
    'BC': 'C1',
    'DAgger': 'C3',
    'Hybrid_2': 'C9',
    'Hybrid_8': 'C2',
    'Hybrid_32': 'C4',
}

x_vals = (np.arange(1, total_rounds + 1) * horizon)
x_labels = [f"{x // 1000}k" for x in x_vals]

def bootstrap_confidence_bounds(data, n_bootstrap=1000, alpha=0.2):
    data = np.array(data)

    # If scalar or single value
    if data.ndim == 0 or len(data) == 1:
        return float(data), float(data)

    bootstrap_means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(bootstrap_means, 100 * (alpha / 2))
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return lower, upper

plt.figure(figsize=(16, 5))

# === Subplot 1: Performance ===
plt.subplot(1, 3, 1)
for name in metrics:
    costs = metrics[name]['cost']  # shape: (rounds, seeds)
    y, lb, ub = [], [], []
    for round_vals in costs:
        vals = 100 - np.array(round_vals)  # convert cost to performance
        mean = np.mean(vals)
        l, u = bootstrap_confidence_bounds(vals)
        y.append(mean)
        lb.append(l)
        ub.append(u)
    plt.plot(x_vals, y, label=name_mapping[name], color=color_mapping[name])
    plt.fill_between(x_vals, lb, ub, color=color_mapping[name], alpha=0.2)
plt.title('Policy Performance')
plt.xlabel('Number of Expert Anntations')
plt.ylabel('Test Reward')
plt.xticks(x_vals[::30], x_labels[::30])
# plt.ylim(-0.05, 1.05)
plt.grid(True)
# plt.legend()

# === Subplot 2: N0 Coverage ===
plt.subplot(1, 3, 2)
for name in metrics:
    coverages = metrics[name]['n0_coverage']  # shape: (rounds, seeds)
    y, lb, ub = [], [], []
    for round_vals in coverages:
        vals = np.array(round_vals)
        mean = np.mean(vals)
        l, u = bootstrap_confidence_bounds(vals)
        y.append(mean)
        lb.append(l)
        ub.append(u)
    plt.plot(x_vals, y, label=name_mapping[name], color=color_mapping[name])
    plt.fill_between(x_vals, lb, ub, color=color_mapping[name], alpha=0.2)
plt.title('Average E State Coverage')
plt.xlabel('Number of Expert Anntations')
plt.ylabel('Coverage Rate')
plt.xticks(x_vals[::30], x_labels[::30])
plt.ylim(-0.05, 1.05)
plt.grid(True)
# plt.legend()

# === Subplot 3: N1 Coverage ===
plt.subplot(1, 3, 3)
for name in metrics:
    coverages = metrics[name]['n1_coverage']  # shape: (rounds, seeds)
    y, lb, ub = [], [], []
    for round_vals in coverages:
        vals = np.array(round_vals)
        mean = np.mean(vals)
        l, u = bootstrap_confidence_bounds(vals)
        y.append(mean)
        lb.append(l)
        ub.append(u)
    plt.plot(x_vals, y, label=name_mapping[name], color=color_mapping[name])
    plt.fill_between(x_vals, lb, ub, color=color_mapping[name], alpha=0.2)
plt.title('Average E\' State Coverage')
plt.xlabel('Number of Expert Anntations')
plt.ylabel('Coverage Rate')
plt.xticks(x_vals[::30], x_labels[::30])
plt.ylim(-0.05, 1.05)
plt.grid(True)
# plt.legend()

plt.tight_layout()
plt.show()
