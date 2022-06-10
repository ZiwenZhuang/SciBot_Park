# SciBot Park
A collection of simulation environment that potentially lead to AGI

## Some advantages of this project
1. You don't need to install all dependencies to run each simulation
2. You can run multiple simulations in parallel (deepcopy is not recommended)

## How to use this project
1. Clone the repository
2. Run `pip install .` in your virtual environment
3. Load modules in your python file
``` Python
import scibotpark, gym
env = gym.make("PandaManipulation-v0")
```
    You can check `scibotpark.__init__.py` for all well implemented environment.

## How to develop based on this project
1. Clone the repository
2. Run `pip install -e .` in your virtual environment
3. Check `scibotpark.__init__.py` for all well implemented environment and inherit from them.
