"""
Franka Robot Control with Recording and Playback

Controls:
- Z: Toggle recording of joint positions
- X: Toggle playback of recorded trajectory
- C: Play S-shaped trajectory and auto-record joint positions
- RIGHT/LEFT: Move target position (Y axis, left/right)
- UP/DOWN: Move target position (X axis, forward/backward)
- Q/E: Rotate target (yaw)
- W/S: Move target (Z axis, up/down)

The robot will:
1. Follow the target cube via inverse kinematics in normal mode
2. Record joint positions when Z is pressed
3. Replay the recorded trajectory when X is pressed
4. Auto-record joint positions when following an S-trajectory (C key)
"""

# ============================================================================
# IMPORTS
# ============================================================================

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import carb
import csv
import os
from datetime import datetime
from pathlib import Path

from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import KinematicsSolver
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget
from isaacsim.core.utils.numpy import rotations as rot_utils
import omni.appwindow as appwindow

# ============================================================================
# HELPER CLASSES
# ============================================================================

class PlaybackAction:
    """Wrapper class for applying recorded joint positions to the robot."""
    
    def __init__(self, joint_positions):
        self.joint_positions = joint_positions
        self.joint_indices = None      # Use all joints
        self.joint_velocities = None   # Use default velocity control
        self.joint_efforts = None      # Use default effort control
    
    def get_length(self):
        """Return the number of joints."""
        return len(self.joint_positions)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_s_trajectory(start_pos, scale=0.15, num_points=200):
    """Generate an S-shaped trajectory from top to bottom.
    
    Args:
        start_pos: Starting position [x, y, z]
        scale: Scale factor for the trajectory amplitude (larger = bigger movement)
        num_points: Number of points in the trajectory
        
    Returns:
        List of position arrays defining the S-shaped path moving downward
    """
    trajectory = []
    for i in range(num_points):
        # Normalize time from 0 to 1
        t = i / (num_points - 1) if num_points > 1 else 0
        
        # S-curve: from top to bottom with S-shaped side motion
        x_offset = scale * 1.0 * np.sin(np.pi * t)  # S-shaped horizontal variation
        y_offset = -scale * 1.2 * t                  # Downward movement
        z_offset = 0                                  # Keep Z constant
        
        pos = np.array([
            start_pos[0] + x_offset,
            start_pos[1] + y_offset,
            start_pos[2] + z_offset
        ])
        trajectory.append(pos)
    
    return trajectory


def load_latest_csv(output_dir):
    """Load the most recent CSV file from the output directory.
    
    Args:
        output_dir: Directory containing recorded CSV files
        
    Returns:
        List of numpy arrays, each containing joint positions for one frame
    """
    try:
        csv_files = [
            f for f in os.listdir(output_dir)
            if f.startswith('joint_positions_') and f.endswith('.csv')
        ]
        
        if not csv_files:
            carb.log_warn(f"No CSV files found in {output_dir}")
            return []
        
        latest_file = sorted(csv_files)[-1]
        csv_path = os.path.join(output_dir, latest_file)
        carb.log_info(f"Loading: {latest_file}")
        
        loaded_data = []
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                if row:
                    # Skip frame number and parse all recorded joint columns.
                    joint_positions = [float(x) for x in row[1:] if x != '']
                    loaded_data.append(np.array(joint_positions))
        
        carb.log_info(f"Loaded {len(loaded_data)} frames from CSV")
        return loaded_data
        
    except Exception as e:
        carb.log_warn(f"Error loading CSV: {e}")
        return []


def save_joint_data_to_csv(recorded_data, output_dir):
    """Save recorded joint data to a CSV file with timestamp.
    
    Args:
        recorded_data: List of joint position arrays
        output_dir: Directory to save the CSV file
        
    Returns:
        Filename of saved CSV, or None if save failed
    """
    if not recorded_data:
        return None
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(output_dir, f"joint_positions_{timestamp}.csv")
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Keep the first 7 joints as J0..J6 for dataset conversion compatibility.
            # Any extra joints (e.g. gripper/finger) are also saved and ignored by converter.
            n_joints = len(recorded_data[0])
            header = ['Frame'] + [f'J{i}' for i in range(min(7, n_joints))]
            if n_joints > 7:
                header += [f'Extra{i}' for i in range(n_joints - 7)]
            writer.writerow(header)
            # Write data rows
            for frame_idx, frame_data in enumerate(recorded_data):
                writer.writerow([frame_idx] + list(frame_data))
        
        carb.log_info(f"Recording saved to: {csv_filename}")
        return csv_filename
        
    except Exception as e:
        carb.log_warn(f"Error saving CSV: {e}")
        return None


# ============================================================================
# INITIALIZATION
# ============================================================================

# ----
# World and Robot Setup
# ----

my_world = World(stage_units_in_meters=1.0)
my_task = FollowTarget(name="follow_target_task")
my_world.add_task(my_task)
my_world.reset()

task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]

my_franka = my_world.scene.get_object(franka_name)
my_controller = KinematicsSolver(my_franka)
articulation_controller = my_franka.get_articulation_controller()
target_obj = my_world.scene.get_object(target_name)

# ----
# Keyboard Setup
# ----

input = None
keyboard = None
keyboard_enabled = False

try:
    input = carb.input.acquire_input_interface()
    app_win = appwindow.get_default_app_window()
    if input is not None:
        keyboard_enabled = True
        keyboard = app_win.get_keyboard()
        print("Keyboard input enabled")
    else:
        print("Keyboard input not available")
except Exception as e:
    carb.log_warn(f"Failed to initialize keyboard input: {e}")

# ----
# Control Parameters
# ----

move_scale = 0.01
yaw_scale = move_scale * 2.0
z_scale = move_scale

# ----
# Recording/Playback Setup
# ----

repo_root = Path(__file__).resolve().parents[1]
output_dir = str(repo_root / "datasets" / "isaac_sim_data")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    carb.log_info(f"Created output directory: {output_dir}")

# Recording state
is_recording = False
recorded_data = []
previous_record_button_state = False

# Playback state
is_playing = False
playback_data = []
playback_frame = 0
previous_play_button_state = False

# Trajectory state
is_following_trajectory = False
trajectory_data = []
trajectory_frame = 0
trajectory_start_pos = None
trajectory_start_joint_positions = None  # Store initial robot joint positions
reset_robot_to_start = False  # Flag to reset robot after trajectory ends
reset_robot_count = 0  # Counter for reset duration
previous_x_button_state = False

# Simulation state
reset_needed = False


# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

while simulation_app.is_running():
    my_world.step(render=True)
    
    # Handle world reset
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
        
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False

        # ====================================================================
        # KEYBOARD INPUT HANDLING
        # ====================================================================
        
        if keyboard_enabled and input is not None and keyboard is not None:
            try:
                # ---- Play Trajectory Button (C key) ----
                c_button_pressed = (
                    input.get_keyboard_value(keyboard, carb.input.KeyboardInput.C) > 0.5
                )
                
                # Detect button press (rising edge)
                if c_button_pressed and not previous_x_button_state:
                    if not is_following_trajectory:
                        # Save current robot joint positions as starting point.
                        obs = my_world.get_observations()
                        trajectory_start_joint_positions = obs[franka_name]["joint_positions"].copy()

                        # Start a new S-shaped trajectory.
                        trajectory_start_pos = target_obj.get_world_pose()[0].copy()
                        trajectory_data = generate_s_trajectory(trajectory_start_pos, scale=0.5, num_points=200)
                        trajectory_frame = 0
                        is_following_trajectory = True
                        
                        # Auto-start recording
                        is_recording = True
                        recorded_data = []
                        carb.log_info(">>> S-TRAJECTORY STARTED & RECORDING <<<")
                    else:
                        # Stop trajectory
                        is_following_trajectory = False
                        carb.log_info("Trajectory stopped")
                
                previous_x_button_state = c_button_pressed

                # ---- Record Button (Z key) ----
                record_button_pressed = (
                    input.get_keyboard_value(keyboard, carb.input.KeyboardInput.Z) > 0.5
                )
                
                # Detect button press (rising edge)
                if record_button_pressed and not previous_record_button_state:
                    is_recording = not is_recording
                    if is_recording:
                        recorded_data = []
                        carb.log_info(">>> RECORDING STARTED <<<")
                    else:
                        save_joint_data_to_csv(recorded_data, output_dir)
                
                previous_record_button_state = record_button_pressed

                # ---- Playback Button (X key) ----
                play_button_pressed = (
                    input.get_keyboard_value(keyboard, carb.input.KeyboardInput.X) > 0.5
                )
                
                # Detect button press (rising edge)
                if play_button_pressed and not previous_play_button_state:
                    is_playing = not is_playing
                    if is_playing:
                        playback_data = load_latest_csv(output_dir)
                        playback_frame = 0
                        if playback_data:
                            carb.log_info(f">>> PLAYBACK STARTED ({len(playback_data)} frames) <<<")
                        else:
                            is_playing = False
                    else:
                        carb.log_info("Playback stopped")
                
                previous_play_button_state = play_button_pressed

                # ---- Target Control (Keyboard) - Only when not playing and not following trajectory ----
                if not is_playing and not is_following_trajectory:
                    # Get keyboard input values
                    left = input.get_keyboard_value(keyboard, carb.input.KeyboardInput.LEFT)
                    right = input.get_keyboard_value(keyboard, carb.input.KeyboardInput.RIGHT)
                    up = input.get_keyboard_value(keyboard, carb.input.KeyboardInput.UP)
                    down = input.get_keyboard_value(keyboard, carb.input.KeyboardInput.DOWN)
                    w = input.get_keyboard_value(keyboard, carb.input.KeyboardInput.W)
                    s = input.get_keyboard_value(keyboard, carb.input.KeyboardInput.S)
                    q = input.get_keyboard_value(keyboard, carb.input.KeyboardInput.Q)
                    e = input.get_keyboard_value(keyboard, carb.input.KeyboardInput.E)
                    
                    # Calculate movement values
                    lx = right - left        # LEFT/RIGHT for Y axis
                    ly = up - down           # UP/DOWN for X axis
                    rx = e - q               # Q/E for yaw rotation
                    ry = w - s               # W/S for Z axis

                    # Update target position and orientation
                    cur_pos, cur_ori = target_obj.get_world_pose()
                    
                    # Position updates
                    cur_pos[0] += ly * move_scale      # X axis (UP/DOWN)
                    cur_pos[1] -= lx * move_scale      # Y axis (LEFT/RIGHT)
                    cur_pos[2] += ry * z_scale         # Z axis (W/S)

                    # Rotation update (yaw only)
                    euler = rot_utils.quats_to_euler_angles(cur_ori)
                    yaw = float(euler[2]) + rx * yaw_scale
                    new_quat = rot_utils.euler_angles_to_quats(np.array([np.pi, 0.0, yaw]))
                    
                    target_obj.set_world_pose(position=cur_pos, orientation=new_quat)
                    
            except Exception as e:
                carb.log_warn(f"Keyboard input error: {e}")

        # ====================================================================
        # TRAJECTORY FOLLOWING
        # ====================================================================
        
        if is_following_trajectory and trajectory_data:
            if trajectory_frame < len(trajectory_data):
                # Update target to the next point in the trajectory
                cur_ori = target_obj.get_world_pose()[1]
                target_obj.set_world_pose(position=trajectory_data[trajectory_frame], orientation=cur_ori)
                trajectory_frame += 1
            else:
                # Trajectory completed - reset robot and target to starting positions
                is_following_trajectory = False
                is_recording = False
                save_joint_data_to_csv(recorded_data, output_dir)
                
                # Reset target_obj to starting position
                if trajectory_start_pos is not None:
                    cur_ori = target_obj.get_world_pose()[1]
                    target_obj.set_world_pose(position=trajectory_start_pos, orientation=cur_ori)
                
                # Flag to reset robot to starting joint positions
                reset_robot_to_start = True
                reset_robot_count = 0
                
                carb.log_info(">>> S-TRAJECTORY COMPLETED & SAVED <<<")
                carb.log_info(">>> ROBOT RESETTING TO START POSITION <<<")

        # ====================================================================
        # ROBOT CONTROL
        # ====================================================================
        
        observations = my_world.get_observations()
        target_position = observations[target_name]["position"]
        target_orientation = observations[target_name]["orientation"]

        if reset_robot_to_start and trajectory_start_joint_positions is not None:
            # Reset robot to starting joint positions for up to 50 frames
            if reset_robot_count < 50:
                action = PlaybackAction(trajectory_start_joint_positions)
                articulation_controller.apply_action(action)
                reset_robot_count += 1
            else:
                reset_robot_to_start = False
                carb.log_info(">>> ROBOT RESET COMPLETE - READY FOR NEXT TRAJECTORY <<<")
        elif is_playing and playback_data:
            # Playback mode: apply recorded joint positions
            if playback_frame < len(playback_data):
                action = PlaybackAction(playback_data[playback_frame])
                articulation_controller.apply_action(action)
                playback_frame += 1
            else:
                # Playback finished
                is_playing = False
                carb.log_info("Playback completed")
        else:
            # Normal mode: use inverse kinematics
            actions, succ = my_controller.compute_inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
            )
            if succ:
                articulation_controller.apply_action(actions)

        # ====================================================================
        # DATA RECORDING
        # ====================================================================
        
        joint_positions = observations[franka_name]["joint_positions"]
        if is_recording:
            recorded_data.append(list(joint_positions))

simulation_app.close()
