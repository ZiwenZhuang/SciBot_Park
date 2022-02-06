# SciBot Park
A collection of simulation environment that potentially lead to AGI

## Some advantages of this project
1. You don't need to install all dependencies to run each simulation
2. You can run multiple simulations in parallel (deepcopy is not recommended)

## How to use this project
1. Clone the repository
2. Run `pip install -e .` in your virtual environment
3. Load modules in your python file
``` Python
from scibotpark.unitree.example_env import ExampleEnv # only pybullet needs to be installed
```