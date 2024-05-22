# Fast Obstacle Avoidance for Dynamic Environments
---
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
---

## Introduction
This work describes a Fast Obstacles Avoidance (FOA) which can be used, when the known input data is sampled (unstructured), or in the presence of many obstacles.
The algorithm is able to input sampled data, as well as a desired velocity, and outputs the collision-free velocity.

<p align="center">
<img src="https://github.com/hubernikus/fast_obstacle_avoidance/blob/main/media/single_obstacle_avoidance_sampled.gif?raw=true"  width="550"></>

Since the input to the algorithm is raw sensor data (in the cartesian frame), it can directly be applied to an arbitrary number of data points. 
<p align="center">
<img src="https://github.com/hubernikus/fast_obstacle_avoidance/blob/main/media/multi_obstacle_avoidance_sampled.gif?raw=true"  width="550"></>

Improved performance can be achieved by first clustering the data points with commonly used techniques, here we use DBSCAN.
<p align="center">
<img src="https://github.com/hubernikus/fast_obstacle_avoidance/blob/main/media/two_obstacle_avoidance_clustersampled.gif?raw=true"  width="550"></>

A video describing the algorithm in depth can be found under:     
[![Alt text](https://img.youtube.com/vi/kr7R_cJoaYI/0.jpg)](https://youtu.be/kr7R_cJoaYI)


### Controller Parameters
The algorithm has several free parameters which can be tuned to adapt the performance based on the environment.

The distance for each data point $o$ is calculated as follows:

$$ 
 \Gamma_o(\xi) = \| \xi - \xi_o \| / R_0
$$

where $R_0$ is the control radius, which accounts for the robot geometry.

Crucial for the function of the algorithm is the  influence weight that each point $o$ has on the agent. This is calculated with this formula:

$$ 	 
\hat{w}_o(\xi) = \left( \frac{D^{\mathrm{scal}}}{D_o(\xi)} \right)^s
	\qquad \text{with} \quad
	D_o (\xi) = \Gamma_o(\xi) - 1
$$

An example script that analyses the different parameters can be found in `./examples/example_parameter_comparison.py`


#### Control Radius
A larger control radius $R_0$ increases the distance at which the obstacle is avoided. This can increase safe operation and take into account a larger robot geometry:

<table>
  <tr>
  <td> <img src="./media/comparison_control_radius_01.svg"> </td>
  <td> <img src="./media/comparison_control_radius_10.svg"> </td>
  <td> <img src="./media/comparison_control_radius_20.svg"> </td>
  </tr>
</table>

<table>
 <tr>
 	<td> $$D_0 = 0.1 $$ </td>
	<td> $$D_0 = 1.0$$ </td>
	<td> $$D_0 = 2.0$$ </td>
  </tr>
</table>

Attribute name: `control_radius`


#### Weight Factor $D^{\mathrm{scal}}$
The weight factor is a simple scaling on the weight. An increased value results in an increased avoidance effect (higher modulation) further away from the obstacle. This results in safer avoidance, but with lower similarity to the original trajectory. 

<!-- | <img src="./media/comparison_weight_factor_1.svg">  | <img src="./media/comparison_weight_factor_3.svg"> | <img src="./media/comparison_weight_factor_10.svg"> | -->
<table>
  <tr>
  <td> <img src="./media/comparison_weight_factor_1.svg"> </td>
  <td> <img src="./media/comparison_weight_factor_3.svg"> </td>
  <td> <img src="./media/comparison_weight_factor_10.svg"> </td>
  </tr>
</table>
<!-- <tr>  -->
<!-- 	<td> </td> -->
<!-- 	<td> </td> -->
<!-- 	<td> </td> -->
<!-- </tr> -->
<table>
 <tr>
	<td> $$D^{\mathrm{scal}} = 1 \delta $$ </td>
	<td> $$D^{\mathrm{scal}} = 3 \delta$$ </td>
	<td> $$D^{\mathrm{scal}} = 10 \delta$$ </td>
  </tr>
</table>

where $\delta = 2 \pi / N^{\mathrm{samples}}$ is the sampling angle, with which the agent observes the space.

Attribute name: `weight_factor`


#### Weight Power $s$
The weight power increases weights above one but decreases weights below zero. Note, that most weights have a value below one. Hence, with an increased scaling value, the effect on the surroundings is lower. However, when getting close to the obstacle this effect is inverted (as we get weights larger than one).

<!-- | <img src="./fast_obstacle_avoidance/media/comparison_weight_power_10.svg">  | <img src="./fast_obstacle_avoi -->
<!-- dance/media/comparison_weight_power_15.svg"> | <img src="./fast_obstacle_avoidance/media/comparison_weight_power_20.svg"> | -->
<!-- |:---:|:---:|:---:| -->
<!-- | $s = 1.0$ | $s = 1.5$ | $s = 2.0$ | -->
<table>
  <tr>
  <td> <img src="./media/comparison_weight_power_10.svg"> </td>
  <td> <img src="./media/comparison_weight_power_15.svg"> </td>
  <td> <img src="./media/comparison_weight_power_20.svg"> </td>
  </tr>
</table>

<table>
 <tr>
 	<td> $$D_0 = 0.1 $$ </td>
	<td> $$D_0 = 1.0$$ </td>
	<td> $$D_0 = 2.0$$ </td>
  </tr>
</table>


Attribute name: `weight_power`

# Create Custom Python Environment

## Installation / Setup
To set it, go to your install/code directory, and type:
```sh
git clone --recurse-submodules https://github.com/hubernikus/fast_obstacle_avoidance.git
```
(Make sure submodules are there if various_tools library is not installed.)

Go to the file directory:
```sh
cd fast_obstacle_avoidance
``` 

Setup your environment, with Python>3.7 (here Python 3.9):
Choose your favorite python environment. I recommend using [virtual environment venv](https://docs.python.org/3/library/venv.html).
Set up the virtual environment (use whatever compatible environment manager that you have with python -V >= 3.10).

``` bash
python3.10 -m venv .venv
```

Activate your environment
``` sh
source .venv/bin/activate
```

### Setup Dependencies
Install all requirements:
``` bash
pip install -r requirements.txt && python setup.py develop
```

#### Install dependencies
This package depends on two libraries:
Dynamic obstacle avoidance
https://github.com/hubernikus/dynamic_obstacle_avoidance.git

Various Tools
https://github.com/hubernikus/various_tools.git

Make sure you have these installed, otherwise, this can be done by:

``` sh
mkdir src && cd src
git clone git@github.com:hubernikus/dynamic_obstacle_avoidance.git && cd dynamic_obstacle_avoidance
pip install -r requirements.txt && python setup.py develop && cd ..
https://github.com/hubernikus/various_tools.git && cd various_tools
pip install -r requirements.txt && python setup.py develop && cd ../..
```

## Get Started
To get started, try to run the files in the 'examples' folder.

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

## Issues / Improvements
- The current repository does not have any test for CI/CD. This will be added if desired for future uses.
- The clustering is based on DBSCAN (which has a simple hyperparameter choice), however, it does not scale well with the number of dimensions or the number of datapoints. Faster approaches should be adapted to be useful for the current method.


## Debug
You forgot to add the submodules, add them with:
``` sh
git submodule update --init --recursive
```

## Citing Repository
If you use this repository in a scientific publication, please use the following citation:

Huber, Lukas. _Exact Obstacle Avoidance for Robots in Complex and Dynamic Environments Using Local Modulation._ No. 10373., EPFL, 2024.

Bibtex entry:
``` bibtex
@phdthesis{huber2024exact,
  title={Exact Obstacle Avoidance for Robots in Complex and Dynamic Environments Using Local Modulation},
  author={Huber, Lukas},
  year={2024},
  month={April},
  address={Lausanne, Switzerland},
  school={EPFL},
  type={PhD thesis}
}
```

(c) hubernikus
