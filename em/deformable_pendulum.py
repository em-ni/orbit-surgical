# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a deformable pendulum.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_deformable_pendulum.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on simulating a deformable pendulum.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import carb
from carb.input import KeyboardInput

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Ceiling plane
    cfg = RigidObjectCfg(
        prim_path="/World/Ceiling",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    ceiling = RigidObject(cfg=cfg)

    # Deformable Cylinder
    cfg = DeformableObjectCfg(
        prim_path="/World/Cylinder",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.05,
            height=1.0,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e6),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 1.45)),
        debug_vis=True,
    )
    cylinder_object = DeformableObject(cfg=cfg)

    # return the scene information
    scene_entities = {"cylinder_object": cylinder_object, "ceiling": ceiling}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    cylinder_object = entities["cylinder_object"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    print("\n[INFO]: Cylinder length will vary sinusoidally during simulation.")

    # Get the default nodal state of the object
    initial_nodal_state = cylinder_object.data.default_nodal_state_w.clone()
    default_nodal_state = cylinder_object.data.default_nodal_state_w.clone()
    nodal_pos_w = default_nodal_state[0, :, :3]

    # Find top vertices to constrain
    top_z = torch.max(nodal_pos_w[:, 2])
    top_indices = torch.where(nodal_pos_w[:, 2] > top_z - 0.01)[0]

    # Set up kinematic targets
    nodal_kinematic_target = cylinder_object.data.nodal_kinematic_target.clone()
    # Free all vertices initially
    nodal_kinematic_target[0, :, 3] = 1.0
    # Constrain the top vertices
    nodal_kinematic_target[0, top_indices, 3] = 0.0
    # Set their target position to their initial position
    nodal_kinematic_target[0, top_indices, :3] = nodal_pos_w[top_indices]

    # Write kinematic target to simulation
    cylinder_object.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

    # Simulate physics
    while simulation_app.is_running():
        # Apply sinusoidal length variation
        scale_factor = 1.0 + 0.3 * torch.sin(torch.tensor(sim_time * 2.0))  # Varies between 0.7 and 1.3

        # Update current state with scaling
        current_nodal_state = initial_nodal_state.clone()
        current_nodal_state[0, :, 2] *= scale_factor

        # Write states to simulation
        cylinder_object.write_nodal_state_to_sim(current_nodal_state)

        # write internal data to simulation
        cylinder_object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        cylinder_object.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 1.0])
    # Design scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
