#from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch
import sys
sys.path.append('/workspaces/humanoid_g1/unitree_sdk2_python')
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data,quaternion_to_euler_array
from common.remote_controller import RemoteController, KeyMap
from config import Config
import math
import os
from collections import deque
ROOT_DIR = os.environ.get("PROJECT_ROOT", "/workspaces/humanoid_g1")
import onnx
import onnxruntime as ort

class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)

        model = onnx.load("/home/irev/deploy_mujoco/deploy/policy_1_21dof_test.onnx")
        onnx.checker.check_model(model)
        self.policy = ort.InferenceSession("/home/irev/deploy_mujoco/deploy/policy_1_21dof_test.onnx")
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.hist_obs = deque(maxlen=9)
        # self.buffer_action=np.zeros((3,12),dtype=np.float32)


        for _ in range(config.num_obs_frame - 1):
            self.hist_obs.append(np.zeros([config.num_obs], dtype=np.float32))
        for _ in range(10):
            action = self.policy.run(None, {'input': np.zeros([1, config.num_obs * config.num_obs_frame], dtype=np.float32)})[0]
        # for _ in range(config.num_obs_frame):
        #     self.hist_obs.append(np.zeros([1, config.num_obs], dtype=np.float32))

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self, dry_run=False):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        
        # eu_ang = quaternion_to_euler_array(quat)
        # eu_ang[eu_ang > math.pi] -= 2 * math.pi

        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = self.config.num_actions
        num_obs=self.config.num_obs
        # obs[:1] = sin_phase
        # obs[1:2] = cos_phase
        # obs[2:5] = cmd * cmd_scale
        # obs[5:5 + num_actions] = qj
        # obs[5 + num_actions:5 + num_actions * 2] = dqj
        # obs[5 + num_actions * 2:5 + num_actions * 3] = action
        # obs[5 + num_actions * 3:8 + num_actions * 3] = omega
        # obs[8 + num_actions * 3:11 + num_actions * 3] = gravity_orientation
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.obs[:1]=sin_phase
        self.obs[1:2]=cos_phase
        self.obs[2:5]= self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[5:5 + num_actions]=qj_obs
        self.obs[5 + num_actions:5 + num_actions * 2]=dqj_obs
        self.obs[5 + num_actions * 2:5 + num_actions * 3]=self.action#*0.5
        self.obs[5 + num_actions * 3:8 + num_actions * 3]=ang_vel
        self.obs[8 + num_actions * 3:11 + num_actions * 3]=gravity_orientation#eu_ang
        
        # self.obs[:3] = ang_vel
        # self.obs[3:6] = gravity_orientation
        # self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        # self.obs[9 : 9 + num_actions] = qj_obs
        # self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        # self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        # self.obs[9 + num_actions * 3] = sin_phase
        # self.obs[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        
        # obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        # self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        obs_copy= np.clip(self.obs, -18, 18)




        obs_hist = np.array(self.hist_obs).flatten()

        self.hist_obs.append(obs_copy)
        # hist_obs.popleft()
        policy_input = np.concatenate([obs_hist, obs_copy])[None, :]
        # policy_input = torch.from_numpy(policy_input).float().unsqueeze(0)
        # obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        # policy inference
        self.action = self.policy.run(None, {'input': policy_input})[0]

        # self.hist_obs.append(self.obs)
        # self.hist_obs.popleft()
        #
        # policy_input = np.zeros([1,self.config.num_obs_frame*self.config.num_obs], dtype=np.float32)
        # for i in range(self.config.num_obs_frame):
        #     policy_input[0, i * self.config.num_obs : (i + 1) * self.config.num_obs] =self.hist_obs[i]
        # #obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        # # policy inference
        # self.action = self.policy(torch.tensor(policy_input)).detach().numpy().squeeze()
        self.action = np.clip(self.action, -4., 4.)
        
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale
        if not dry_run:
            
            # Build low cmd
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            # send the command
            self.send_cmd(self.low_cmd)

            time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1_ours.yaml")
    args = parser.parse_args()

    # Load config
    config_path =f"{ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)
    print(config.policy_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()
    
    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()
    for _ in range(10):
        controller.run(dry_run=True)
    controller.counter=0
    print("dddddddddddddrrrrry_run")
    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
