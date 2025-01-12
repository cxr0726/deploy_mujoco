import time

import mujoco.viewer
import mujoco
import numpy as np
import os

ROOT_DIR = os.environ.get("PROJECT_ROOT", "/home/lc/copy")
import torch
import yaml
from collections import deque
import math
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
   # x, y, z, w = quat
    
    w,x,y,z=quat
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}",ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        num_obs_frame=config["frame"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)
    hist_obs = deque()
    print("wwwwwww")
    for _ in range(num_obs_frame):
        hist_obs.append(np.zeros([1, num_obs], dtype=np.float32))
    for _ in range(10):
        action = policy(torch.zeros(1,470)).detach().numpy().squeeze()
    import mujoco_viewer
    viewer = mujoco_viewer.MujocoViewer(m, d)
    if True:
    #with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.
                t1=time.time()
                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                print(gravity_orientation,"ggg",omega)
                omega = omega * ang_vel_scale
                # eu_ang = quaternion_to_euler_array(quat)
                # eu_ang[eu_ang > math.pi] -= 2 * math.pi
               
                period = 0.8
                count = counter * simulation_dt
                phase = count / period#% period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)


                obs[:1]=sin_phase
                obs[1:2]=cos_phase
                obs[2:5]= cmd * cmd_scale
                obs[5:17]=qj
                obs[17:29]=dqj
                obs[29:41]=action
                obs[41:44]=omega
                obs[44:47]=gravity_orientation
                

                
                # obs[:3] = omega
                # obs[3:6] = gravity_orientation
                # obs[6:9] = cmd * cmd_scale
                # obs[9 : 9 + num_actions] = qj
                # obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                # obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                # obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                
                obs_copy = np.clip(obs, -18, 18)

                hist_obs.append(obs_copy)
                hist_obs.popleft()

                policy_input = np.zeros([1,num_obs_frame*num_obs], dtype=np.float32)
                for i in range(num_obs_frame):
                    policy_input[0, i * num_obs : (i + 1) * num_obs] = hist_obs[i]
                #obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(torch.tensor(policy_input)).detach().numpy().squeeze()
                action = np.clip(action, -4., 4.)
                #print(action)
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
                t2=time.time()
                print(t2-t1)
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            #viewer.sync()
            viewer.render()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
