This is the codebase used to create and run the optimization problems for the thesis "How to Ride a Bicycle Without Hands"
by Simon Sorgedrager, made at the Bicycle Lab at the faculty Mechanical Engineering of the TU Delft.

The .yml file has the dependencies needed to run this, except for that BRiM (symBRiM) has to be installed separately via
https://github.com/mechmotum/symbrim.


The code content of this project is divided into two parts:
 - BRiM model files
 - Code to construct the model, run optimizations and create plots

Let's discuss the BRiM model files first.
The __init__.py file was intented to ease the way the models can be called upon by the other
model files, and by the optimization scripts. However this gave some loop-import errors so it ended up not being used.

As is the convention in BRiM, the different models have base from which the parameters and required or dependant models are defined.
There are three types of files here, that either describe; body models, connection models, models that consist of multiple bodies and connections, and the base of those bodies and connections.

All the bases of the bodies are put together in the SimBodyBase.py file, where each of these bases is created by me, and different than the similar bases if they already exist in BRiM, because the model used in this project has different properties, as is the case for the Torso for example.
The connection bases also share a file in SimJointBase.py, though this only covers the neck joint, and the sliding seat mechanism joints.
The model files of the bodies are also grouped in Sim_bodies.py (apologies for the inconsistency in naming, it also bothers me). This covers the three bodies of the rider, and the intermediate seat body for the sliding seat mechanism.
The actual connection model files are grouped together per joint (sim_seats, SimTorsoJoints, SimNeckJoints) as there can be multiple different joints (i.e. pin, spherical and weld for the torso). Also, the torque actuation and spring-damper functions are added in these files.

The other files describe the models. These are the rider (simrider.py), bicycle-rider (sim_bicycle_rider.py), and a bicycle model (WhippleBicycle_Sprung_Steering.py).
The bicycle model is only created to incorporate a spring between the front- and rear-frame. Also, sim_rear_frame.py is created to facilitate this spring.

Now let's discuss the optimization files.

The optimizations are run from opt_run.py. This file houses all the arguments of the argument_parser. These arguments
define the model configuration (rider bodies, joints, torques and tasks) which are selected from Enum classes, and add optimization parameters (which are defined somewhere else) to the system.
The file calls all functions from the other files and logs them.

opt_main.py defines the optimization parameters like duration, timestep, and distances of the trajectory paths.

The model is set up in opt_model.py. Here, all the features required to build the bicycle-rider model from symbrim and
the model files in this repository are imported, after which the model is created. Firstly the model is built, after which
spring-dampers and joint Torques are added. Then, the system is verified with the generalized coordinates which creates
the holonomic loop constraints. At the bottom of opt_model.py, the simulator is set up.

All the constraints (initial, final and the system bounds) are defined in opt_problem.py, with the different task constraints
defined with if-else statements. Before setting these constraints, the initial guess of the optimization is created by either running the simulation
set up in opt_model.py, or by giving different arrays to the optimizer. After the problem is set up, it is solved by calling
the data.problem.solve function.

The functions required by this simulator are stored in opt_simulator, nothing has to be changed here in order to run
or change the optimization.

opt_utils.py houses many functions, like a check_config function that verifies if the choses model configuration in opt_run.py
makes sense, or the function that creates the objective function of the problem object. opt_utils also houses all the
functions that creates plots and animations after each optimization.

opt_container.py contains the Enum classes, the datastorage class, and creates the variable functions as the state vector x
and input vector r. It also contains data of the path trajectories.

opt_data_plotter.py has the functions that are used to create different plots which are used in the thesis report.
It has a function that can plot data from different optimizations together, or from individual optimizations.
Other specific functions to plot data can be found here, like a function to create stability plots.

