# RoboCasa-v0.2 DAgger

This is a DAgger-ready fork of [**Robocasa v0.2**](https://github.com/robocasa/robocasa) [10/31/2024]: using RoboSuite `v1.5` as the backend [**[Paper]**](https://robocasa.ai/assets/robocasa_rss24.pdf). 

-------
## Key DAgger updates
Two saving modes are supported:
- **Intervention-only saving**: only the timesteps where the human intervened (i.e., corrected the robot policy) are saved. This makes it easy to inspect where and how often the policy fails, and to build datasets focused on corrective behavior.
- **Full-rollout saving**: the entire episode is saved regardless of intervention frequency. This captures blended behavior — segments of autonomous policy execution interleaved with human corrections — which is useful for training on the full distribution of states the robot encounters during deployment, with markers denoting acting agents.

-------
## Installation
For standalone installation:
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n robocasa python=3.10
   ```
2. Activate conda environment:
   ```sh
   conda activate robocasa
   ```
3. Clone and setup robosuite dependency (**important: use the master branch!**):

   ```sh
   git clone https://github.com/ARISE-Initiative/robosuite
   cd robosuite
   pip install -e .
   ```
4. Clone and setup this repo:

   ```sh
   cd ..
   git clone https://github.com/robocasa/robocasa
   cd robocasa
   pip install -e .
   pip install pre-commit; pre-commit install           # Optional: set up code formatter.

   (optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)
   ```
5. Install the package and download assets:
   ```sh
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
   python robocasa/scripts/setup_macros.py              # Set up system variables.
   ```
