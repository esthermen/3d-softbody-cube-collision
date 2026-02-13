3D Soft-Body Simulation with Rigid Collision
Overview

This project implements a 3D deformable soft-body cube using a mass-spring lattice model.

The cube interacts with a rigid planar obstacle including restitution and friction.

The system includes:

Structural springs

Shear springs

Bend springs

Gravity

Damping

Collision detection and response

Integration is performed using a global RK4 scheme.

Physical Components
Spring Types

Structural springs

Shear springs

Bend springs

External Forces

Gravity

Damping

Collision Model

Rigid plane

Coefficient of restitution

Coulomb friction approximation

Numerical Method

Fourth-order Runge-Kutta

Dependencies
numpy
matplotlib


Install:

pip install -r requirements.txt


Run:

python main.py

Author

Esther Menéndez
Physics-based Simulation & Numerical Methods

## Simulation Preview

![SoftBody Simulation](media/softbody_simulation.gif)

