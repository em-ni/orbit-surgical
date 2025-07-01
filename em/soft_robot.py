# How to run:
# %IsaacLab_PATH%\isaaclab.bat -p soft_robot.py

# Isaac Sim imports should come first before any other heavy imports
# import isaacsim  # Import isaacsim first to avoid deprecation warning
from isaacsim import SimulationApp

# Initialize SimulationApp first with proper settings
simulation_app = SimulationApp(
    {
        "headless": False,
        "physics_enabled": True,
        "renderer_enabled": True,
        "width": 1280,
        "height": 720,
    }
)

# Wait for initialization
simulation_app.update()

# Now import other libraries after SimulationApp initialization
import torch
from typing import Optional
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np

# import carb
# import omni.appwindow  # Contains handle to keyboard

print("Libraries imported successfully")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class sfr(nn.Module):
    def __init__(self) -> None:
        super(sfr, self).__init__()
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
        dydt = torch.zeros((batch_size, 14)).to(device)

        e3 = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, 3, 1).repeat(batch_size, 1, 1)
        ux = y[:, 12]
        uy = y[:, 13]

        # Compute u_hat for each batch element
        u_hat = torch.zeros((batch_size, 3, 3), device=device)
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
        t_eval_top = torch.arange(0.0, l_top + self.ds, self.ds).to(device)

        # Solve ODE for top segment only
        sol_top = odeint(self.odeFunction, y0_top.unsqueeze(0), t_eval_top)

        return sol_top

    def downsample_simple(self, arr, m):
        n = len(arr)
        indices = np.linspace(0, n - 1, m, dtype=int)
        return arr[indices]


class Simulation:
    def __init__(self, numb_sphere=15) -> None:
        self.num_sphere = numb_sphere

        # Add robot and device attributes
        self.robot: Optional[sfr] = None
        self.device: Optional[torch.device] = None

        # Add mesh path attributes for deformable body
        self.base_mesh_path = None
        self.deformable_material_path = None

        # Keyboard availability flag
        self._keyboard_available = True

        # Keyboard input throttling
        self._last_key_time = {}
        self._key_throttle_delay = 0.1  # 100ms between key presses

        # Initialize control variables for manual control
        self.manual_elongation = 1.0  # Default elongation factor (1.0 = no elongation)
        self.manual_bend_y = 0.0  # Manual Y-axis bending
        self.manual_bend_x = 0.0  # Manual X-axis bending
        self.control_mode = "auto"  # "auto" or "manual"

        # Transition variables for smooth manual-to-auto switching
        self.auto_start_time = None
        self.manual_to_auto_elongation_offset = 0.0
        self.manual_to_auto_bend_y_offset = 0.0
        self.manual_to_auto_bend_x_offset = 0.0

        # Transition variables for smooth mode switching
        self.auto_start_time = 0.0  # Time when auto mode was last activated
        self.manual_to_auto_elongation_offset = 0.0  # Offset to add to sinusoid for smooth transition
        self.manual_to_auto_bend_y_offset = 0.0  # Y-bend offset for smooth transition
        self.manual_to_auto_bend_x_offset = 0.0  # X-bend offset for smooth transition

        # Keyboard input mapping for soft robot control
        self._input_keyboard_mapping = {
            # Elongation controls
            "Q": ("elongation", 0.1),  # Extend (Q key)
            "A": ("elongation", -0.1),  # Contract (A key)
            # Y-axis bending controls (left/right when viewed from behind)
            "W": ("bend_y", 0.005),  # Bend right (W key)
            "S": ("bend_y", -0.005),  # Bend left (S key)
            # X-axis bending controls (forward/backward)
            "E": ("bend_x", 0.005),  # Bend forward (E key)
            "D": ("bend_x", -0.005),  # Bend backward (D key)
            # Mode switching
            "SPACE": ("mode_toggle", 0),  # Toggle between auto and manual mode
            # Reset controls
            "R": ("reset", 0),  # Reset to neutral position
        }

        # Import Isaac Sim API after SimulationApp is initialized
        from omni.isaac.core import World

        # Create world with specific physics settings for GPU dynamics
        self.my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)

        # Enable GPU dynamics immediately after world creation
        physics_context = self.my_world.get_physics_context()
        physics_context.enable_gpu_dynamics(flag=True)

        self.stage = self.my_world.stage
        self.soft_body_prim = None

        print("World created with GPU dynamics enabled")
        print("\n=== KEYBOARD CONTROLS ===")
        print("Q/A: Extend/Contract hanging robot base (0.3x to 5.0x)")
        print("W/S: Bend hanging segment left/right")
        print("E/D: Bend hanging segment forward/backward")
        print("SPACE: Toggle Auto/Manual mode (smooth transition)")
        print("R: Reset to neutral position")
        print("=========================")
        print("Note: Robot hangs from ceiling and grows downward.")
        print("When switching modes, the robot will")
        print("smoothly transition over 5 seconds.")
        print("If standard keyboard controls don't work, polling method will be used.")

    def create_robot(self):
        # Import objects after world is created
        from omni.isaac.core.objects import VisualCylinder, VisualSphere
        import omni.kit.commands
        from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

        global simulation_app

        # Create physics scene with GPU dynamics
        physics_scene_path = "/World/PhysicsScene"

        if not self.stage.GetPrimAtPath(physics_scene_path):
            omni.kit.commands.execute("AddPhysicsScene", stage=self.stage, path=physics_scene_path)

        physics_scene_prim = self.stage.GetPrimAtPath(physics_scene_path)
        if physics_scene_prim:
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
            physx_scene_api.CreateEnableGPUDynamicsAttr().Set(True)
            physx_scene_api.CreateSolverTypeAttr().Set("TGS")
            physx_scene_api.CreateBroadphaseTypeAttr().Set("GPU")
            print("Physics scene configured for GPU dynamics")

        # Force updates
        for _ in range(3):
            simulation_app.update()

        # Create a ceiling anchor point (small sphere at the ceiling)
        self.ceiling_height = 0.4  # 40cm ceiling height
        self.ceiling_anchor = VisualSphere(
            prim_path="/World/CeilingAnchor",
            name="ceiling_anchor",
            position=np.array([0, 0, self.ceiling_height]),  # At ceiling
            radius=0.01,  # 10mm radius
            color=np.array([128, 128, 128]),  # Gray color
        )
        self.my_world.scene.add(self.ceiling_anchor)

        # Create VISUAL base segment as a cylinder that dangles from ceiling (BLUE section in sketch)
        # This will be kinematically controlled for elongation and hangs down from ceiling
        initial_cylinder_height = 0.05  # 50mm initial height
        cylinder_center_z = self.ceiling_height - initial_cylinder_height / 2.0  # Hangs down from ceiling

        self.base_cylinder = VisualCylinder(
            prim_path="/World/BaseCylinder",
            name="base_cylinder",
            position=np.array([0, 0, cylinder_center_z]),  # Hangs from ceiling
            radius=0.003,  # 3mm radius (thinner)
            height=initial_cylinder_height,  # 50mm initial height
            color=np.array([0, 0, 255]),  # Blue color
        )
        self.my_world.scene.add(self.base_cylinder)

        self.base_cylinder_path = "/World/BaseCylinder"
        self.ceiling_anchor_path = "/World/CeilingAnchor"
        self.base_height_original = 0.05

        print(f"Ceiling anchor created at height {self.ceiling_height}m")
        print("Vertical base cylinder created (blue section - hangs from ceiling, kinematically controlled elongation)")

        # Create spheres for the top kinematic segment only (RED section in sketch)
        # This is the bending segment that hangs below the blue cylinder
        initial_sphere_start_z = self.ceiling_height - initial_cylinder_height  # Bottom of blue cylinder

        for i in range(self.num_sphere):
            sphere_z = initial_sphere_start_z - i * 0.003  # Hang down from bottom of blue cylinder
            self.my_world.scene.add(
                VisualSphere(
                    prim_path="/sphere" + str(i),
                    name="visual_sphere" + str(i),
                    position=np.array([0, 0, sphere_z]),  # Hang below the base cylinder
                    radius=0.003,  # 3mm radius (thicker, matches cylinder)
                    color=(
                        np.array([255, 0, 0]) if i != self.num_sphere - 1 else np.array([0, 255, 0])
                    ),  # Red with green tip
                )
            )

    def reset(self):
        self.my_world.scene.add_default_ground_plane()
        self.my_world.reset()

        # Set up keyboard listener
        self.setup_keyboard()

        # Note: Deformable body creation is commented out as it's not needed for this kinematic simulation
        # The soft robot is simulated using kinematic control of visual elements

        # Enable gravity
        physics_context = self.my_world.get_physics_context()
        physics_context.set_gravity(-9.81)
        print("Gravity enabled")

    def get_soft_body_top_position(self):
        """Get the position of the top of the VERTICAL soft body"""
        if self.soft_body_prim:
            try:
                from pxr import UsdGeom, Gf

                # Get mesh geometry
                mesh_geom = UsdGeom.Mesh(self.soft_body_prim)
                if mesh_geom:
                    points_attr = mesh_geom.GetPointsAttr()
                    if points_attr:
                        points = points_attr.Get()
                        if points and len(points) > 0:
                            # Find highest Z coordinate (top of vertical cylinder)
                            max_z = max(point[2] for point in points)
                            top_points = [point for point in points if abs(point[2] - max_z) < 0.001]

                            if top_points:
                                # Calculate center of top points
                                center_x = sum(p[0] for p in top_points) / len(top_points)
                                center_y = sum(p[1] for p in top_points) / len(top_points)
                                center_z = max_z

                                # Get world transform
                                xformable = UsdGeom.Xformable(self.soft_body_prim)
                                transform_matrix = xformable.ComputeLocalToWorldTransform(0)

                                # Transform to world coordinates
                                local_top = Gf.Vec3d(center_x, center_y, center_z)
                                world_top = transform_matrix.Transform(local_top)

                                return np.array([world_top[0], world_top[1], world_top[2]])

                # Fallback: estimate top position
                xformable = UsdGeom.Xformable(self.soft_body_prim)
                if xformable:
                    transform_matrix = xformable.ComputeLocalToWorldTransform(0)
                    translation = transform_matrix.ExtractTranslation()
                # Final fallback - use current elongation factor if available
                try:
                    original_height = 0.05
                    if "elongation_factor" in locals():
                        estimated_height = original_height * elongation_factor
                    else:
                        estimated_height = original_height
                    top_position = np.array([translation[0], translation[1], translation[2] + estimated_height])
                except:
                    top_position = np.array([translation[0], translation[1], translation[2] + 0.05])
                return top_position

            except Exception as e:
                print(f"Error getting soft body top position: {e}")

        # Final fallback
        return np.array([0, 0, 0.05])  # 5cm above ground

    def setup_keyboard(self):
        """Set up keyboard listener for manual control"""
        try:
            import carb.input
            import omni.appwindow

            self._appwindow = omni.appwindow.get_default_app_window()

            # Try different methods to get the input interface for Isaac Sim 4.1.0
            try:
                # Method 1: Modern carb interface
                self._input = carb.get_framework().get_interface(carb.input.IInput)
            except:
                try:
                    # Method 2: Legacy acquire method
                    self._input = carb.input.acquire_input_interface()
                except:
                    # Method 3: Direct instantiation
                    self._input = carb.input.IInput()

            self._keyboard = self._appwindow.get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
            print("Keyboard controls activated!")
            self._keyboard_available = True

        except Exception as e:
            print(f"Warning: Could not set up keyboard controls: {e}")
            print("Trying alternative input method...")

            # Alternative method using Isaac Sim's input system
            try:
                import omni.kit.actions.core

                self._action_registry = omni.kit.actions.core.get_action_registry()
                print("Alternative input method activated! Use the following keys:")
                print("  - Press 'M' to toggle manual/auto mode")
                print("  - Manual controls will be active when in manual mode")
                self._keyboard_available = True
            except Exception as e2:
                print(f"Alternative input method failed: {e2}")
                print("Manual controls will not be available. The simulation will run in auto mode only.")
                self._keyboard_available = False

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        """
        Keyboard subscriber callback for soft robot control
        """
        # Check if keyboard controls are available
        if not getattr(self, "_keyboard_available", False):
            return True

        import carb.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name

            if key_name in self._input_keyboard_mapping:
                action_type, value = self._input_keyboard_mapping[key_name]

                if action_type == "mode_toggle":
                    # Toggle between auto and manual mode
                    old_mode = self.control_mode
                    self.control_mode = "manual" if self.control_mode == "auto" else "auto"

                    # If switching from manual to auto, calculate smooth transition offsets
                    if old_mode == "manual" and self.control_mode == "auto":
                        # Store the current simulation time and manual values to create smooth transition
                        if hasattr(self.my_world, "current_time"):
                            current_time = self.my_world.current_time
                            self.auto_start_time = current_time

                            # Calculate what the auto mode sinusoids would be at this time
                            w_bend = 3 * np.pi
                            auto_elongation = 1.0 + 1.5 * np.sin(0.5 * w_bend * current_time)
                            auto_elongation = max(0.5, min(3.0, auto_elongation))  # Apply same clamping as auto mode
                            auto_bend_y = 0.012 * np.sin(w_bend * current_time)  # Use updated amplitude
                            auto_bend_x = 0.0

                            # Calculate offsets to make auto mode start from current manual values
                            self.manual_to_auto_elongation_offset = self.manual_elongation - auto_elongation
                            self.manual_to_auto_bend_y_offset = self.manual_bend_y - auto_bend_y
                            self.manual_to_auto_bend_x_offset = self.manual_bend_x - auto_bend_x

                            print(
                                f"Smooth transition: elongation offset {self.manual_to_auto_elongation_offset:.3f}, bend_y offset {self.manual_to_auto_bend_y_offset*1000:.1f}mm"
                            )
                        else:
                            # Fallback if time not available
                            self.auto_start_time = 0.0
                            self.manual_to_auto_elongation_offset = self.manual_elongation - 1.0
                            self.manual_to_auto_bend_y_offset = self.manual_bend_y
                            self.manual_to_auto_bend_x_offset = self.manual_bend_x

                    elif old_mode == "auto" and self.control_mode == "manual":
                        # When switching from auto to manual, update manual values to match current auto state
                        if hasattr(self.my_world, "current_time"):
                            current_time = self.my_world.current_time
                            w_bend = 3 * np.pi

                            # Calculate current auto values including any remaining transition effects
                            time_since_auto_start = current_time - self.auto_start_time
                            decay_factor = max(0.0, 1.0 - time_since_auto_start / 5.0)

                            base_elongation = 1.0 + 1.5 * np.sin(0.5 * w_bend * current_time)
                            base_elongation = max(0.5, min(3.0, base_elongation))
                            base_bend_y = 0.012 * np.sin(w_bend * current_time)
                            base_bend_x = 0.0

                            # Include any remaining transition offsets
                            current_auto_elongation = (
                                base_elongation + self.manual_to_auto_elongation_offset * decay_factor
                            )
                            current_auto_bend_y = base_bend_y + self.manual_to_auto_bend_y_offset * decay_factor
                            current_auto_bend_x = base_bend_x + self.manual_to_auto_bend_x_offset * decay_factor

                            # Set manual values to match current auto state
                            self.manual_elongation = current_auto_elongation
                            self.manual_bend_y = current_auto_bend_y
                            self.manual_bend_x = current_auto_bend_x

                            print(
                                f"Synced manual values: elongation {self.manual_elongation:.3f}, bend_y {self.manual_bend_y*1000:.1f}mm"
                            )

                    print(f"Control mode switched to: {self.control_mode.upper()}")

                elif action_type == "reset":
                    # Reset to neutral position
                    self.manual_elongation = 1.0
                    self.manual_bend_y = 0.0
                    self.manual_bend_x = 0.0

                    # Reset transition offsets
                    self.manual_to_auto_elongation_offset = 0.0
                    self.manual_to_auto_bend_y_offset = 0.0
                    self.manual_to_auto_bend_x_offset = 0.0

                    print("Robot reset to neutral position")
                    if self.control_mode == "manual":
                        self.update_visual_realtime()  # Update visuals immediately

                elif self.control_mode == "manual":
                    # Only allow manual control in manual mode
                    if action_type == "elongation":
                        self.manual_elongation = max(0.3, min(50.0, self.manual_elongation + value))
                        print(f"Elongation: {self.manual_elongation:.2f}x")
                        self.update_visual_realtime()  # Update visuals immediately

                    elif action_type == "bend_y":
                        self.manual_bend_y = max(-0.02, min(0.02, self.manual_bend_y + value))
                        print(f"Y-Bend: {self.manual_bend_y*1000:+.1f}mm")
                        self.update_visual_realtime()  # Update visuals immediately

                    elif action_type == "bend_x":
                        self.manual_bend_x = max(-0.02, min(0.02, self.manual_bend_x + value))
                        print(f"X-Bend: {self.manual_bend_x*1000:+.1f}mm")
                        self.update_visual_realtime()  # Update visuals immediately

                else:
                    if action_type in ["elongation", "bend_y", "bend_x"]:
                        print("Manual control disabled. Press SPACE to enable manual mode.")

        return True

    def check_keyboard_input(self):
        """Simple keyboard polling method as fallback"""
        if not self._keyboard_available:
            return

        try:
            # Try to check keyboard state using Isaac Sim's input system
            import omni.appwindow
            import carb.input

            appwindow = omni.appwindow.get_default_app_window()
            keyboard = appwindow.get_keyboard()

            # Check for key presses manually (using int values for key states)
            if keyboard.get_key_state(carb.input.KeyboardInput.Q) > 0:
                self.handle_key_input("elongation", 0.1)
            elif keyboard.get_key_state(carb.input.KeyboardInput.A) > 0:
                self.handle_key_input("elongation", -0.1)
            elif keyboard.get_key_state(carb.input.KeyboardInput.W) > 0:
                self.handle_key_input("bend_y", 0.005)
            elif keyboard.get_key_state(carb.input.KeyboardInput.S) > 0:
                self.handle_key_input("bend_y", -0.005)
            elif keyboard.get_key_state(carb.input.KeyboardInput.E) > 0:
                self.handle_key_input("bend_x", 0.005)
            elif keyboard.get_key_state(carb.input.KeyboardInput.D) > 0:
                self.handle_key_input("bend_x", -0.005)
            elif keyboard.get_key_state(carb.input.KeyboardInput.SPACE) > 0:
                self.handle_key_input("mode_toggle", 0)
            elif keyboard.get_key_state(carb.input.KeyboardInput.R) > 0:
                self.handle_key_input("reset", 0)

        except Exception as e:
            pass  # Silent fail for polling method

    def handle_key_input(self, action_type, value):
        """Handle keyboard input actions with throttling"""
        import time

        current_time = time.time()

        # Throttle key presses to prevent too rapid input
        if action_type in self._last_key_time:
            if current_time - self._last_key_time[action_type] < self._key_throttle_delay:
                return

        self._last_key_time[action_type] = current_time

        if action_type == "mode_toggle":
            # Toggle between auto and manual mode
            old_mode = self.control_mode
            self.control_mode = "manual" if self.control_mode == "auto" else "auto"
            print(f"Control mode switched to: {self.control_mode.upper()}")

        elif action_type == "reset":
            # Reset to neutral position
            self.manual_elongation = 1.0
            self.manual_bend_y = 0.0
            self.manual_bend_x = 0.0
            print("Robot reset to neutral position")
            if self.control_mode == "manual":
                self.update_visual_realtime()

        elif self.control_mode == "manual":
            # Only allow manual control in manual mode
            if action_type == "elongation":
                self.manual_elongation = max(0.3, min(50.0, self.manual_elongation + value))
                print(f"Elongation: {self.manual_elongation:.2f}x")
                self.update_visual_realtime()

            elif action_type == "bend_y":
                self.manual_bend_y = max(-0.02, min(0.02, self.manual_bend_y + value))
                print(f"Y-Bend: {self.manual_bend_y*1000:+.1f}mm")
                self.update_visual_realtime()

            elif action_type == "bend_x":
                self.manual_bend_x = max(-0.02, min(0.02, self.manual_bend_x + value))
                print(f"X-Bend: {self.manual_bend_x*1000:+.1f}mm")
                self.update_visual_realtime()
        else:
            if action_type in ["elongation", "bend_y", "bend_x"]:
                print("Manual control disabled. Press SPACE to enable manual mode.")

    def update_visual_realtime(self):
        """Update visual elements immediately when manual control values change"""
        if self.control_mode != "manual":
            return

        if not hasattr(self, "robot") or not hasattr(self, "device") or self.robot is None or self.device is None:
            return  # Robot and device not yet set

        try:
            # Update cylinder height and position for hanging configuration
            elongation_factor = self.manual_elongation
            base_cylinder = self.my_world.scene.get_object("base_cylinder")
            if base_cylinder:
                # Calculate new height and position (hanging from ceiling)
                original_height = 0.05  # 50mm original height
                new_height = original_height * elongation_factor

                # Position cylinder so its top stays at ceiling and it grows downward
                new_position = np.array([0, 0, self.ceiling_height - new_height / 2.0])  # Hangs from ceiling

                # Update cylinder height and position
                from pxr import UsdGeom, Gf

                # Get the cylinder prim
                cylinder_prim = self.stage.GetPrimAtPath(self.base_cylinder_path)
                if cylinder_prim:
                    # Update height attribute
                    cylinder_geom = UsdGeom.Cylinder(cylinder_prim)
                    cylinder_geom.GetHeightAttr().Set(new_height)

                    # Update position safely
                    xformable = UsdGeom.Xformable(cylinder_prim)
                    if xformable:
                        # Clear existing transform ops to avoid conflicts
                        xformable.ClearXformOpOrder()
                        # Add new translation
                        translate_op = xformable.AddTranslateOp()
                        translate_op.Set(Gf.Vec3d(new_position[0], new_position[1], new_position[2]))

            # Update sphere positions for top segment (hanging below the blue cylinder)
            bottom_of_cylinder = self.ceiling_height - 0.05 * elongation_factor
            base_bottom_position = np.array([0, 0, bottom_of_cylinder])

            # Create actions for robot simulation (hanging configuration)
            length_change = self.robot.l0_base * (elongation_factor - 1.0)
            actions = torch.tensor([length_change, self.manual_bend_y, self.manual_bend_x], device=self.device)

            # Update top segment starting from bottom of hanging cylinder
            y0_top = self.robot.y0.clone()
            y0_top[0:3] = torch.tensor(base_bottom_position, device=self.device)

            # Set bending parameters
            l_base, ux_base, uy_base, l_top, ux_top, uy_top = self.robot.updateAction(actions)
            y0_top[12] = ux_top
            y0_top[13] = uy_top

            # Solve ODE for top segment (bending segment hangs down)
            t_eval_top = torch.arange(0.0, l_top + self.robot.ds, self.robot.ds).to(self.device)
            sol_top = odeint(self.robot.odeFunction, y0_top.unsqueeze(0), t_eval_top)

            # Update sphere positions for hanging top segment
            # Extract position data from the ODE solution
            if isinstance(sol_top, torch.Tensor):
                sol_top_positions = sol_top[:, 0, :3]  # Extract positions from solution tensor
                sol_top_downsampled = self.robot.downsample_simple(sol_top_positions, self.num_sphere)
            else:
                # If sol_top is a tuple/list, use the first element
                sol_top_tensor = sol_top[0] if isinstance(sol_top, (list, tuple)) else sol_top
                sol_top_positions = sol_top_tensor[:, 0, :3]
                sol_top_downsampled = self.robot.downsample_simple(sol_top_positions, self.num_sphere)

            if isinstance(sol_top_downsampled, torch.Tensor):
                sol_top_downsampled = sol_top_downsampled.detach().cpu().numpy()

            for i in range(self.num_sphere):
                sphere = self.my_world.scene.get_object("visual_sphere" + str(i))
                if sphere:
                    # For hanging configuration, we need to flip the Z direction since robot hangs down
                    if len(sol_top_downsampled.shape) == 3:
                        ode_position = sol_top_downsampled[i, 0, :]
                    else:
                        ode_position = sol_top_downsampled[i, :]

                    # The ODE gives positions assuming upward growth, but we want downward hanging
                    # So we flip the Z component relative to the starting position
                    hanging_position = np.array(
                        [
                            base_bottom_position[0]
                            + ode_position[0]
                            - base_bottom_position[0],  # X: keep relative offset
                            base_bottom_position[1]
                            + ode_position[1]
                            - base_bottom_position[1],  # Y: keep relative offset
                            base_bottom_position[2]
                            - (ode_position[2] - base_bottom_position[2]),  # Z: flip direction for hanging
                        ]
                    )

                    sphere.set_world_pose(position=hanging_position)

        except Exception as e:
            print(f"Error in real-time visual update: {e}")


# Create robot and simulation instances
robot = sfr().to(device)
sim = Simulation(numb_sphere=15)
sim.robot = robot  # Store robot reference in simulation
sim.device = device  # Store device reference in simulation
sim.create_robot()
sim.reset()

# Wait for simulation to be ready
simulation_app.update()

# Start simulation
sim.my_world.play()

step_counter = 0

print("Starting simulation loop...")
while simulation_app.is_running():
    if sim.my_world.is_playing():
        # Check for keyboard input (fallback method)
        sim.check_keyboard_input()

        w_bend = 3 * np.pi
        current_time = sim.my_world.current_time

        # Use manual control values if in manual mode, otherwise use automatic control
        if sim.control_mode == "manual":
            # Manual control mode: use keyboard-controlled values
            elongation_factor = sim.manual_elongation
            bend_y_top = sim.manual_bend_y
            bend_x_top = sim.manual_bend_x
        else:
            # Automatic control mode: use sinusoidal patterns with smooth transition offsets
            w_bend = 3 * np.pi

            # Base sinusoidal values with increased elongation range
            elongation_amplitude = 0.02  # 20mm elongation amplitude
            base_elongation = 1.0 + 1.5 * np.sin(
                0.5 * w_bend * current_time
            )  # Vary between -0.5x and 2.5x (0.5x to 2.5x clamped)
            base_elongation = max(0.5, min(3.0, base_elongation))  # Clamp to reasonable range

            bend_amplitude = 0.012  # Increased bend amplitude to 12mm
            base_bend_y = bend_amplitude * np.sin(w_bend * current_time)
            base_bend_x = 0.0

            # Add transition offsets for smooth mode switching (these decay over time)
            time_since_auto_start = current_time - sim.auto_start_time
            decay_factor = max(0.0, 1.0 - time_since_auto_start / 5.0)  # Decay over 5 seconds

            elongation_factor = base_elongation + sim.manual_to_auto_elongation_offset * decay_factor
            bend_y_top = base_bend_y + sim.manual_to_auto_bend_y_offset * decay_factor
            bend_x_top = base_bend_x + sim.manual_to_auto_bend_x_offset * decay_factor

        # Update the blue cylinder's height and position dynamically (hanging from ceiling)
        base_cylinder = sim.my_world.scene.get_object("base_cylinder")
        if base_cylinder:
            try:
                # Calculate new height and position for hanging configuration
                original_height = 0.05  # 50mm original height
                new_height = original_height * elongation_factor

                # Position cylinder so its top stays at ceiling and it grows downward
                new_position = np.array([0, 0, sim.ceiling_height - new_height / 2.0])  # Hangs from ceiling

                # Update cylinder height and position
                from pxr import UsdGeom, Gf

                # Get the cylinder prim
                cylinder_prim = sim.stage.GetPrimAtPath(sim.base_cylinder_path)
                if cylinder_prim:
                    # Update height attribute
                    cylinder_geom = UsdGeom.Cylinder(cylinder_prim)
                    cylinder_geom.GetHeightAttr().Set(new_height)

                    # Update position safely
                    xformable = UsdGeom.Xformable(cylinder_prim)
                    if xformable:
                        # Clear existing transform ops to avoid conflicts
                        xformable.ClearXformOpOrder()
                        # Add new translation
                        translate_op = xformable.AddTranslateOp()
                        translate_op.Set(Gf.Vec3d(new_position[0], new_position[1], new_position[2]))

            except Exception as e:
                print(f"Error updating cylinder: {e}")

        # Calculate the bottom position of the hanging blue cylinder (where red segment should start)
        bottom_of_cylinder = sim.ceiling_height - 0.05 * elongation_factor
        base_bottom_position = np.array([0, 0, bottom_of_cylinder])

        # Create actions for robot simulation (hanging configuration)
        # Convert elongation factor to absolute length change in meters
        length_change = robot.l0_base * (elongation_factor - 1.0)  # Absolute change from base length
        actions = torch.tensor([length_change, bend_y_top, bend_x_top], device=device)

        # Update top segment starting from bottom of hanging cylinder
        y0_top = robot.y0.clone()
        y0_top[0:3] = torch.tensor(base_bottom_position, device=device)

        # Set bending parameters
        l_base, ux_base, uy_base, l_top, ux_top, uy_top = robot.updateAction(actions)
        y0_top[12] = ux_top
        y0_top[13] = uy_top

        # Solve ODE for top segment (hanging down from blue cylinder)
        t_eval_top = torch.arange(0.0, l_top + robot.ds, robot.ds).to(device)
        sol_top = odeint(robot.odeFunction, y0_top.unsqueeze(0), t_eval_top)

        # Update sphere positions for hanging top segment
        # Extract position data from the ODE solution
        if isinstance(sol_top, torch.Tensor):
            sol_top_positions = sol_top[:, 0, :3]  # Extract positions from solution tensor
            sol_top_downsampled = robot.downsample_simple(sol_top_positions, sim.num_sphere)
        else:
            # If sol_top is a tuple/list, use the first element
            sol_top_tensor = sol_top[0] if isinstance(sol_top, (list, tuple)) else sol_top
            sol_top_positions = sol_top_tensor[:, 0, :3]
            sol_top_downsampled = robot.downsample_simple(sol_top_positions, sim.num_sphere)

        if isinstance(sol_top_downsampled, torch.Tensor):
            sol_top_downsampled = sol_top_downsampled.detach().cpu().numpy()

        for i in range(sim.num_sphere):
            sphere = sim.my_world.scene.get_object("visual_sphere" + str(i))
            if sphere:
                # For hanging configuration, we need to flip the Z direction since robot hangs down
                if len(sol_top_downsampled.shape) == 3:
                    ode_position = sol_top_downsampled[i, 0, :]
                else:
                    ode_position = sol_top_downsampled[i, :]

                # Calculate the bottom of the cylinder for this frame
                bottom_of_cylinder = sim.ceiling_height - 0.05 * elongation_factor
                base_bottom_position = np.array([0, 0, bottom_of_cylinder])

                # The ODE gives positions assuming upward growth, but we want downward hanging
                # So we flip the Z component relative to the starting position
                hanging_position = np.array(
                    [
                        base_bottom_position[0] + ode_position[0] - base_bottom_position[0],  # X: keep relative offset
                        base_bottom_position[1] + ode_position[1] - base_bottom_position[1],  # Y: keep relative offset
                        base_bottom_position[2]
                        - (ode_position[2] - base_bottom_position[2]),  # Z: flip direction for hanging
                    ]
                )

                sphere.set_world_pose(position=hanging_position)

        # Step simulation
        sim.my_world.step(render=True)
        step_counter += 1

        # Print status every 100 steps
        if step_counter % 100 == 0:
            print(f"Simulation step: {step_counter}, Time: {current_time:.3f}")
            print(f"  Control Mode: {sim.control_mode.upper()}")
            print(
                f"  Base segment: Hanging from ceiling at {sim.ceiling_height}m - Elongation factor: {elongation_factor:.2f}x"
            )
            print(f"  Top segment: {robot.l0_top*1000:.1f}mm (kinematic bending, hangs below)")
            print(f"  Y-Bend: {bend_y_top*1000:+.1f}mm, X-Bend: {bend_x_top*1000:+.1f}mm")

    # Update the app
    simulation_app.update()

print("Simulation ended")
simulation_app.close()
