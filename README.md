# fast_obstacle_avoidance
Dynamic but Fast Obstacle Avoidance


# Install
Setup your environment, with Python>3.7 (here Python 3.9):
``` bash
python3.9 -m venv .venv
source .venv/bin/activate
```

# Setup
Install ROS-bag evaluation.
``` bash
pip install -r requirements_ros.txt
pip install rosbag --extra-index-url https://rospypi.github.io/simple/
```

## Requirements
Various Tools [Algebra & Dynamical Systems]
https://github.com/hubernikus/various_tools


# Development Setup
Install the dev-requirements
``` bash
pip install -r requirements_dev.txt
```

In case of using the pre-commit hook, do once:

``` bash
pre-commit install
```

