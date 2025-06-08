
## 🐳 Environment Setup (via Docker)

Launch the Docker container with GPU support:

```bash
docker run --gpus all -it \
  -v ~/workspace \
  -v ~/.mujoco:/root/.mujoco \
  --name exp \
  tensorflow/tensorflow:1.13.2-gpu-py3 bash
```

### Inside the container

1. Install MuJoCo and dependencies:

```bash
apt-get update
apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3
```

2. Set MuJoCo environment variables:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
export MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mjpro150
export MJKEY_PATH=/root/.mujoco/mjkey.txt
```

3. Install Python dependencies:

```bash
pip install 'imageio<2.10.0' 'Cython<3'
pip install mujoco-py==1.50.1.0
pip install mpi4py
pip install torch
```

## 🚀 Running Experiments and plotting

- For multi-run experiments:

```bash
python run_experiment.py
```

- For single-run test (e.g., test policy `a`):

```bash
python soil_function.py
```

- To plot evaluation metrics based on saved results:

```bash
python plot.py
```
