# Fast Obstacle Avoidance for Dynamic Environments

A video describing the algorithm can be found here:   

[![Alt text](https://img.youtube.com/vi/WKso-wu68v8/0.jpg)](https://youtu.be/kr7R_cJoaYI)

# Create Custom Python Environment

## Setup
To setup got to your install/code directory, and type:
```sh
git clone --recurse-submodules https://github.com/hubernikus/fast_obstacle_avoidance.git
```
(Make sure submodules are there if various_tools librarys is not installed.)

Go to file directory:
```sh
cd fast_obstacle_avoidance
``` 

Setup your environment, with Python>3.7 (here Python 3.9):
Choose your favorite python-environment. I recommend to use [virtual environment venv](https://docs.python.org/3/library/venv.html).
Setup virtual environment (use whatever compatible environment manager that you have with Python >3.7).

``` bash
python3.10 -m venv .venv
```
with python -V > 3.7

Activate your environment
``` sh
source .venv/bin/activate
```

# Setup Dependencies
Install all requirements:
``` bash
pip install -r requirements.txt && python setup.py develop
```

Install submodules:
``` bash
cd src/various_tools && pip install -r requirements.txt && python setup.py develop && cd ../..
cd src/dynamic_obstacle_avoidance && pip install -r requirements.txt && python setup.py develop && cd ../..
```

# Evaluation of ROS-bags
Install ROS-bag evaluation.
``` bash
pip install -r requirements_ros.txt
pip install rosbag --extra-index-url https://rospypi.github.io/simple/
```

## Requirements
Various Tools [Algebra & Dynamical Systems], 
https://github.com/hubernikus/various_tools


# Development Setup
Install the dev-requirements
``` bash
pip install -r requirements_dev.txt
```

Try to use pre-commit-hooks if actively contributing to the repository:
``` bash
pre-commit install
```

# Debug
You forgot to add the submodules, add them with:
``` sh
git submodule update --init --recursive
```


**References**     
> [1] Huber, Lukas, Aude Billard, and Jean-Jacques E. Slotine. "Avoidance of Convex and Concave Obstacles with Convergence ensured through Contraction." IEEE Robotics and Automation Letters (2019).  

> [2] Huber, Lukas, and Slotine Aude Billard. "Avoiding Dense and Dynamic Obstacles in Enclosed Spaces: Application to Moving in a Simulated Crowd." arXiv preprint arXiv:2105.11743 (2021).  

**Contact**: [Lukas Huber] (https://people.epfl.ch/lukas.huber?lang=en) (lukas.huber AT epfl dot ch)

**Acknowledgments**
This work was funded in part by the SAHR Project.

(c) hubernikus
