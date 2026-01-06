# How to run:
# %IsaacLab_PATH%\isaaclab.bat -p ur5_soft_robot.py

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="ur5", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
from torchdiffeq import odeint
import torch.nn as nn

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_rotate

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from ur5_config import UR5_CFG  # isort:skip


class SoftRobotModel(nn.Module):
    """Soft robot continuum model using ODE integration."""

    def __init__(self, device: torch.device) -> None:
        super(SoftRobotModel, self).__init__()
        self.device = device
        self.l0_base = 50e-3  # initial length of base segment (elongating part)
        self.l0_top = 50e-3  # fixed length of top segment (bending part)
        self.d = 7.5e-3  # cables offset
        self.ds = 0.005  # ode step time

        r0 = torch.zeros(3, 1).to(device)
        R0 = torch.eye(3).reshape(9, 1).to(device)
        y0 = torch.cat((r0, R0, torch.zeros([2, 1], device=device)), dim=0)
        self.y0 = y0.squeeze()

    def updateAction(self, actions):
        # actions: [length_change_base, bend_y_top, bend_x_top]
        # Base segment: only length changes, no bending
        l_base = self.l0_base + actions[0]  # Variable length base
        ux_base = 0.0  # No bending in base segment
        uy_base = 0.0  # No bending in base segment

        # Top segment: fixed length with bending
        l_top = self.l0_top  # Fixed length top segment
        ux_top = actions[2] / -(l_top * self.d)  # Bending in x
        uy_top = actions[1] / (l_top * self.d)  # Bending in y

        return l_base, ux_base, uy_base, l_top, ux_top, uy_top

    def odeFunction(self, s, y):
        batch_size = y.shape[0]
        dydt = torch.zeros((batch_size, 14)).to(self.device)

        e3 = torch.tensor([0.0, 0.0, 1.0], device=self.device).reshape(1, 3, 1).repeat(batch_size, 1, 1)
        ux = y[:, 12]
        uy = y[:, 13]

        # Compute u_hat for each batch element
        u_hat = torch.zeros((batch_size, 3, 3), device=self.device)
        u_hat[:, 0, 2] = uy
        u_hat[:, 1, 2] = -ux
        u_hat[:, 2, 0] = -uy
        u_hat[:, 2, 1] = ux

        r = y[:, 0:3].reshape(batch_size, 3, 1)
        R = y[:, 3:12].reshape(batch_size, 3, 3)

        dR = torch.matmul(R, u_hat)
        dr = torch.matmul(R, e3).squeeze(-1)

        # Reshape and assign to dydt
        dydt[:, 0:3] = dr
        dydt[:, 3:12] = dR.reshape(batch_size, 9)
        return dydt

    def odeStepFull(self, actions):
        # Get segment parameters
        l_base, ux_base, uy_base, l_top, ux_top, uy_top = self.updateAction(actions)

        # Start from end of base segment at the current top position
        y0_top = self.y0.clone()
        y0_top[12] = ux_top  # Add bending to top segment
        y0_top[13] = uy_top  # Add bending to top segment

        # Create time steps for top segment
        t_eval_top = torch.arange(0.0, l_top + self.ds, self.ds).to(self.device)

        # Solve ODE for top segment only
        sol_top = odeint(self.odeFunction, y0_top.unsqueeze(0), t_eval_top)

        return sol_top

    def downsample_simple(self, arr, m):
        n = len(arr)
        indices = np.linspace(0, n - 1, m, dtype=int)
        return arr[indices]


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # Soft robot visual elements - ceiling anchor
    ceiling_anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CeilingAnchor",
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.4)),
    )

    # Soft robot visual elements - base cylinder
    base_cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BaseCylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.003,
            height=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.375)),
    )

    # Deformable cylinder
    deformable_cylinder = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/DeformableCylinder",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.025,
            height=0.3,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.4, youngs_modulus=1e7),
        ),
        # init_state is set dynamically to the robot's end-effector position
        debug_vis=True,
    )

    # Plane attached to end-effector
    ee_plane = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/EEPlane",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),  # Cyan
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur5":
        robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


def create_soft_robot_spheres(scene: InteractiveScene, sim: sim_utils.SimulationContext, num_sphere=15):
    """Create sphere visual elements for the soft robot top segment."""
    spheres = []
    ceiling_height = 0.4
    initial_cylinder_height = 0.05
    initial_sphere_start_z = ceiling_height - initial_cylinder_height

    # Add spheres to the scene configuration dynamically using USD primitives
    from pxr import UsdGeom, Gf

    stage = sim.stage

    # Create a separate visual cylinder that we can control (since RigidObject cylinder can't be easily modified)
    visual_cylinder_path = f"/World/Env_0/VisualCylinder"
    visual_cylinder_prim = UsdGeom.Cylinder.Define(stage, visual_cylinder_path)
    visual_cylinder_prim.GetRadiusAttr().Set(0.003)
    visual_cylinder_prim.GetHeightAttr().Set(initial_cylinder_height)
    visual_cylinder_prim.AddTranslateOp().Set(Gf.Vec3f(0.5, 0.0, ceiling_height - initial_cylinder_height / 2.0))
    visual_cylinder_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 1.0)])  # Blue color

    for i in range(num_sphere):
        sphere_z = initial_sphere_start_z - i * 0.003
        sphere_path = f"/World/Env_0/SoftRobotSphere{i}"

        # Create sphere USD primitive directly
        sphere_prim = UsdGeom.Sphere.Define(stage, sphere_path)
        sphere_prim.GetRadiusAttr().Set(0.003)

        # Set initial position
        sphere_prim.AddTranslateOp().Set(Gf.Vec3f(0.5, 0.0, sphere_z))

        # Set color based on index
        if i != num_sphere - 1:
            # Red color for regular spheres
            sphere_prim.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
        else:
            # Green color for tip sphere
            sphere_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])

        spheres.append(sphere_prim)

    return spheres, visual_cylinder_prim


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]
    ceiling_anchor = scene["ceiling_anchor"]
    base_cylinder = scene["base_cylinder"]
    deformable_cylinder = scene["deformable_cylinder"]
    ee_plane = scene["ee_plane"]

    # Create soft robot model
    device = sim.device
    soft_robot_model = SoftRobotModel(device)

    # Create soft robot sphere elements and visual cylinder
    num_sphere = 15
    ceiling_height = 0.4
    spheres, visual_cylinder = create_soft_robot_spheres(scene, sim, num_sphere)

    # Find the center vertex of the cylinder base to attach
    default_nodes_pos = deformable_cylinder.data.default_nodal_state_w[0, :, :3]
    cylinder_height = 0.3  # From the DeformableObjectCfg
    target_pos = torch.tensor([0.0, 0.0, -cylinder_height / 2.0], device=device)
    distances = torch.linalg.norm(default_nodes_pos - target_pos, dim=1)
    center_vertex_idx = torch.argmin(distances)

    # Nodal kinematic targets of the deformable bodies
    nodal_kinematic_target = deformable_cylinder.data.nodal_kinematic_target.clone()

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
        [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    current_goal_idx = 0
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    elif args_cli.robot == "ur5":
        robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[".*_joint"],
            body_names=["tool0"],  # Correctly target the end-effector link
        )
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    step_counter = 0

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
            count = 0
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

            # Reset deformable cylinder
            deformable_cylinder.reset()
            # Set initial position of the cylinder to the robot's end-effector
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            nodal_state = deformable_cylinder.data.default_nodal_state_w.clone()
            nodal_state[..., :3] = deformable_cylinder.transform_nodal_pos(
                nodal_state[..., :3], ee_pose_w[:, :3], ee_pose_w[:, 3:]
            )
            deformable_cylinder.write_nodal_state_to_sim(nodal_state)

            # Reset spheres to initial positions
            for i, sphere_prim in enumerate(spheres):
                initial_sphere_start_z = ceiling_height - 0.05
                sphere_z = initial_sphere_start_z - i * 0.003
                from pxr import Gf

                sphere_prim.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.5, 0.0, sphere_z))
        else:
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # Attach deformable cylinder to the end-effector
            ee_pos_w = ee_pose_w[:, 0:3]
            nodal_kinematic_target[:, center_vertex_idx, :3] = ee_pos_w
            nodal_kinematic_target[:, center_vertex_idx, 3] = 0.0  # 0: constrained, 1: free
            deformable_cylinder.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)

        # Soft robot simulation
        current_time = count * sim_dt
        w_bend = 3 * np.pi

        # Automatic control mode: use sinusoidal patterns
        base_elongation = 1.0 + 1.5 * np.sin(0.5 * w_bend * current_time)
        base_elongation = max(1, min(3.0, base_elongation))

        bend_amplitude = 0.012
        bend_y_top = bend_amplitude * np.sin(w_bend * current_time)
        bend_x_top = 0.0

        elongation_factor = base_elongation

        # Update the visual cylinder's height and position dynamically using USD primitives
        try:
            original_height = 0.05
            new_height = original_height * elongation_factor
            new_position = np.array([0.5, 0.0, ceiling_height - new_height / 2.0])

            # Update our separate visual cylinder (not the RigidObject one)
            from pxr import UsdGeom, Gf

            stage = sim.stage

            # Update the visual cylinder we created
            if visual_cylinder:
                # Update height attribute
                visual_cylinder.GetHeightAttr().Set(new_height)

                # Update position using USD transform
                existing_ops = visual_cylinder.GetOrderedXformOps()
                translate_op = None

                # Look for existing translate operation
                for op in existing_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break

                # If no translate op exists, create one
                if translate_op is None:
                    translate_op = visual_cylinder.AddTranslateOp()

                # Set the new translation
                translate_op.Set(Gf.Vec3d(new_position[0], new_position[1], new_position[2]))

        except Exception as e:
            print(f"Error updating visual cylinder: {e}")

        # Calculate the bottom position of the hanging blue cylinder
        bottom_of_cylinder = ceiling_height - 0.05 * elongation_factor
        base_bottom_position = np.array([0.5, 0.0, bottom_of_cylinder])

        # Create actions for robot simulation
        length_change = soft_robot_model.l0_base * (elongation_factor - 1.0)
        actions = torch.tensor([length_change, bend_y_top, bend_x_top], device=device)

        # Update top segment starting from bottom of hanging cylinder
        y0_top = soft_robot_model.y0.clone()
        y0_top[0:3] = torch.tensor(base_bottom_position, device=device)

        # Set bending parameters
        l_base, ux_base, uy_base, l_top, ux_top, uy_top = soft_robot_model.updateAction(actions)
        y0_top[12] = ux_top
        y0_top[13] = uy_top

        # Solve ODE for top segment
        t_eval_top = torch.arange(0.0, l_top + soft_robot_model.ds, soft_robot_model.ds).to(device)
        sol_top = odeint(soft_robot_model.odeFunction, y0_top.unsqueeze(0), t_eval_top)

        # Update sphere positions for hanging top segment
        if isinstance(sol_top, torch.Tensor):
            sol_top_positions = sol_top[:, 0, :3]
            sol_top_downsampled = soft_robot_model.downsample_simple(sol_top_positions, num_sphere)
        else:
            sol_top_tensor = sol_top[0] if isinstance(sol_top, (list, tuple)) else sol_top
            sol_top_positions = sol_top_tensor[:, 0, :3]
            sol_top_downsampled = soft_robot_model.downsample_simple(sol_top_positions, num_sphere)

        if isinstance(sol_top_downsampled, torch.Tensor):
            sol_top_downsampled = sol_top_downsampled.detach().cpu().numpy()

        # Update sphere positions using USD primitives
        try:
            from pxr import Gf

            for i in range(num_sphere):
                if i < len(spheres):
                    # For hanging configuration, flip the Z direction since robot hangs down
                    if len(sol_top_downsampled.shape) == 3:
                        ode_position = sol_top_downsampled[i, 0, :]
                    else:
                        ode_position = sol_top_downsampled[i, :]

                    # Calculate hanging position
                    hanging_position = np.array(
                        [
                            base_bottom_position[0] + ode_position[0] - base_bottom_position[0],  # X offset
                            base_bottom_position[1] + ode_position[1] - base_bottom_position[1],  # Y offset
                            base_bottom_position[2]
                            - (ode_position[2] - base_bottom_position[2]),  # Z flipped for hanging
                        ]
                    )

                    # Update sphere position using USD attribute
                    sphere_prim = spheres[i]
                    sphere_prim.GetPrim().GetAttribute("xformOp:translate").Set(
                        Gf.Vec3f(hanging_position[0], hanging_position[1], hanging_position[2])
                    )

        except Exception as e:
            print(f"Error updating sphere positions: {e}")

        # apply actions to UR5
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        deformable_cylinder.write_data_to_sim()
        scene.write_data_to_sim()
        sim.step()
        count += 1
        step_counter += 1
        scene.update(sim_dt)
        deformable_cylinder.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # Update plane pose to follow end-effector with an offset
        plane_offset_local = torch.tensor([[0.0, 0.0, 0.01]], device=device)  # 5cm up
        plane_offset_world = quat_rotate(ee_pose_w[:, 3:7], plane_offset_local)
        plane_pos_w = ee_pose_w[:, 0:3] + plane_offset_world
        plane_pose_w = torch.cat([plane_pos_w, ee_pose_w[:, 3:7]], dim=1)
        ee_plane.write_root_pose_to_sim(plane_pose_w)
        ee_plane.update(sim_dt)
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

        # Print status every 100 steps
        if step_counter % 100 == 0:
            print(f"Simulation step: {step_counter}, Time: {current_time:.3f}")
            print(
                f"  Base segment: Hanging from ceiling at {ceiling_height}m - Elongation factor: {elongation_factor:.2f}x"
            )
            print(f"  Top segment: {soft_robot_model.l0_top*1000:.1f}mm (kinematic bending, hangs below)")
            print(f"  Y-Bend: {bend_y_top*1000:+.1f}mm, X-Bend: {bend_x_top*1000:+.1f}mm")


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    print("[INFO]: UR5 arm will follow predefined waypoints")
    print("[INFO]: Soft robot will hang from ceiling and perform automatic bending/elongation")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
