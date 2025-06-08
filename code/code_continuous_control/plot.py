import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14

# ====== Setup ======
result_folder = ''  # set data folder here
env_list = ['HalfCheetahBulletEnv-v0', 'Walker2DBulletEnv-v0'] #['AntBulletEnv-v0', 'HopperBulletEnv-v0'] 

name_ids = [ 'bc_1', 'dagger_1', 'dagger_1_2', 'dagger_1_4', 'dagger_1_8'] # 'expert',

# Mapping names for legend
name_mapping = {
    'expert': 'Expert',
    'bc_1': 'BC',
    'dagger_1': 'DAgger',
    'dagger_1_2': 'WSD (1/2)',
    'dagger_1_4': 'WSD (1/4)',
    'dagger_1_8': 'WSD (1/8)',
}

# Mapping colors (reuse dagger_1 color)
color_mapping = {
    'expert': 'C0',
    'bc_1': 'C1',
    'dagger_1': 'C3',
    'dagger_1_2': 'C4',
    'dagger_1_4': 'C2',
    'dagger_1_8': 'C9',
}


def bootstrap_confidence_bounds(data, n_bootstrap=1000, alpha=0.2):
    data = np.array(data)
    if data.size == 0:
        print('bootstrap_confidence_bounds: empty data')
        return 0.0, 0.0 
    
    bootstrap_means = [np.mean(np.random.choice(data, len(data))) for _ in range(n_bootstrap)]
    low = np.percentile(bootstrap_means, 100 * (alpha / 2))
    up = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    lower_bound = 2*np.mean(data)-up
    upper_bound = 2*np.mean(data)-low
    return lower_bound, upper_bound

### start plot 

for env_name in env_list:
    plt.figure()

    env_path = os.path.join(os.getcwd(), result_folder, env_name)
    print(env_path)
    for method in name_ids:
        file_path = os.path.join(env_path, f"{method}.pkl")
        color = color_mapping[method]

        with open(file_path, 'rb') as f:
            result = pickle.load(f)

        if method == 'expert':
            mean = np.mean(result['concatenated_result'], axis=0)
            lower, upper = bootstrap_confidence_bounds(result['concatenated_result'])

            plot_size = 20 if env_name in ['HalfCheetahBulletEnv-v0', 'Walker2DBulletEnv-v0'] else 8
            x = result['datasize_list']
            y = np.full((plot_size,), mean)
            lower_bounds = np.full((plot_size,), lower)
            upper_bounds = np.full((plot_size,), upper)
        else:
            print(method)
            mean = np.mean(result['concatenated_result'], axis=1)
            lower_bounds = []
            upper_bounds = []
            for row in result['concatenated_result']:
                lb, ub = bootstrap_confidence_bounds(row)
                lower_bounds.append(lb)
                upper_bounds.append(ub)
            eval_size = len(result['concatenated_result'])

            x = result['datasize_list'][::len(result['datasize_list']) // eval_size]

            y = mean

        plt.plot(x, y, label=name_mapping[method], color=color)
        plt.fill_between(x, lower_bounds, upper_bounds, color=color, alpha=0.2)

    plt.title(env_name.replace('BulletEnv-v0', ''))
    plt.xlabel('Number of Expert Annotations')
    plt.ylabel('Test Reward Value')
    plt.legend()
    plt.grid(True)

    save_dir = os.path.join(env_path, 'overall_plots')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{env_name}.png"))
    plt.show()
    plt.close()