"""RobotRun2 控制器 - 用于达尔文OP2机器人爬梯子的第二阶段（抬腿阶段）."""
import math

#import gym
import time
import numpy as np
import os
import cv2

import argparse
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将ROOT添加到PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径
import torch
from python_scripts.Project_config import gps_goal1


def validate_and_clean_data(data, default_value=0.0):
    """Validate sensor data and replace NaN/Inf values before PPO uses it."""
    if isinstance(data, (list, tuple)):
        return [validate_and_clean_data(x, default_value) for x in data]
    if isinstance(data, np.ndarray):
        return np.nan_to_num(data, nan=default_value, posinf=default_value, neginf=-default_value)
    if isinstance(data, (int, float)):
        if np.isnan(data) or np.isinf(data):
            return default_value
        return data
    return data

class RobotRun2:
    """
    控制机器人按照action行动的类，用于爬梯子阶段
    
    该类负责执行机器人的动作，检测传感器状态，计算奖励，并判断是否完成当前回合
    """
    def __init__(self, robot, state, action_leg_upper, action_leg_lower, action_ankle, step, zhua, gps0, gps1, gps2, gps3, gps4):
        """
        初始化RobotRun2类
        
        参数：
            robot：Webots机器人对象
            state：当前机器人状态（关节角度）
            action_leg_upper：腿部上部舵机动作
            action_leg_lower：腿部下部舵机动作
            action_ankle：脚踝舵机动作
            step：当前步数
            zhua：抓取器状态
            gps0-gps4：GPS传感器数据
        """
        self.robot = robot
        self.timestep = 32  # 仿真时间步长
        self.step = step  # 当前步数
        self.goal = gps_goal1  # 目标位置
        self.biao_zhun = -32.64359902756043  # 标准参考值
        self.robot_state = state  # 当前机器人状态
        # 存储GPS数据
        self.gps0 = gps0
        self.gps1 = gps1
        self.gps2 = gps2
        self.gps3 = gps3
        self.gps4 = gps4
        self.action_leg_upper = action_leg_upper  # 当前动作
        self.action_leg_lower = action_leg_lower  # 当前动作
        self.action_ankle = action_ankle  # 当前动作
        
       
        self.LegUpper = 1.09 * self.action_leg_upper + 0.59
        self.LegLower = 1.14 * self.action_leg_lower - 1.11
        self.Ankle = 1.305 * self.action_ankle - 0.085
        

        self.if_jia = zhua  # 抓取器状态
        self.jie1_Success = False  # 第一阶段是否成功
        self.motors = []  # 电机列表
        self.motors_sensors = []  # 电机传感器列表
        
        # 电机名称列表
        self.motorName = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                     'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
                     'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
                     'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL', 'GraspR')
        
        # 初始化电机和传感器
        for i in range(len(self.motorName)):
            self.motors.append(robot.getDevice(self.motorName[i]))  # 获取电机
            sensorName = self.motorName[i]
            sensorName = sensorName + 'S'  # 传感器名称为电机名称+'S'
            self.motors_sensors.append(self.robot.getDevice(sensorName))  # 获取传感器
            self.motors_sensors[i].enable(self.timestep)  # 启用传感器
            
        # 初始化加速度计和陀螺仪
        self.accelerometer = robot.getDevice('Accelerometer')
        self.gyro = robot.getDevice('Gyro')
        
        # 初始化触摸传感器
        self.touch1 = self.robot.getDevice('touch_foot_L1')
        self.touch1_1 = self.robot.getDevice('touch_foot_L2')
        self.touch1_2 = self.robot.getDevice('touch_foot_L3')
        self.touch5 = self.robot.getDevice('touch_foot_L1')
        self.touch6 = self.robot.getDevice('touch_foot_L2')
        self.touch7 = self.robot.getDevice('touch_foot_R1')
        self.touch8 = self.robot.getDevice('touch_foot_R2')
        self.touch11 = self.robot.getDevice('touch_arm_L1')
        self.touch12 = self.robot.getDevice('touch_arm_R1')
        self.touch13 = self.robot.getDevice('touch_leg_L1')
        self.touch14 = self.robot.getDevice('touch_leg_L2')
        self.touch15 = self.robot.getDevice('touch_leg_R1')
        self.touch16 = self.robot.getDevice('touch_leg_R2')
        
        # 启用触摸传感器
        self.touch1.enable(32)
        self.touch1_1.enable(32)
        self.touch1_2.enable(32)
        self.touch5.enable(32)
        self.touch6.enable(32)
        self.touch7.enable(32)
        self.touch8.enable(32)
        self.touch11.enable(32)
        self.touch12.enable(32)
        self.touch13.enable(32)
        self.touch14.enable(32)
        self.touch15.enable(32)
        self.touch16.enable(32)
        
        # 分组触摸传感器
        self.touch = [self.touch1, self.touch1_1, self.touch1_2]  # 脚部触摸传感器
        self.touch_peng = [self.touch11, self.touch12, self.touch13, self.touch14, self.touch15, self.touch16]  # 碰撞检测触摸传感器
        
        # 初始化状态
        self.future_state = [i for i in self.robot_state]  # 复制当前状态作为未来状态
        
        # 计算下一个状态的关节角度
        self.next = [self.robot_state[11] + self.LegUpper, self.robot_state[13] + self.LegLower, self.robot_state[15] + self.Ankle]

        # 更新未来状态中的关节角度
        self.future_state[11] = self.next[0]  # 左腿上部
        self.future_state[13] = self.next[1]  # 左腿下部
        self.future_state[15] = self.next[2]  # 左脚踝
        print("变化动作")
        print(self.next)

        # 关节角度限制，每个关节的最小和最大角度
        self.limit = [[-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.3], [-2.25, 0.77], [-1.65, 1.16], [-1.18, 1.63],
                      [-2.42, 0.66], [-0.69, 2.5], [-1.01, 1.01], [-1, 0.93], [-1.77, 0.45], [-0.5, 1.68],
                      [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22], [-0.68, 1.04], [-1.02, 0.6],
                      [-1.81, 1.81], [-0.36, 0.94]]
        
        # 初始化当前状态和下一个状态
        self.now_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # 初始化触摸传感器值
        self.touch_value = [0.0, 0.0, 0.0]
        
        # 加速度计和陀螺仪的正常范围
        self.acc_low = [480, 450, 580]  # 加速度计下限
        self.acc_high = [560, 530, 700]  # 加速度计上限
        self.gyro_low = [500, 500, 500]  # 陀螺仪下限
        self.gyro_high = [520, 520, 520]  # 陀螺仪上限

    def run(self):
        """
        执行机器人动作并返回结果
        
        返回：
            tuple：（next_state，reward，done，good，goal，count）
                next_state：下一个状态
                reward：奖励值
                done：是否完成回合
                good：是否为有效动作
                goal：是否达到目标
                count：计数器状态
        """
        self.robot.step(32)  # 执行一个仿真步
        
        # 获取传感器数据
        acc = self.accelerometer.getValues()  # 加速度计数据
        gyro = self.gyro.getValues()  # 陀螺仪数据
        
        # 计算与目标的距离
        x1 = self.goal[0] - self.gps0[1]  # x方向距离
        y1 = self.goal[1] - self.gps0[2]  # y方向距离
        
        # 初始化返回值
        goal = 0  # 是否达到目标
        reward = 0  # 奖励值
        reward1 = math.sqrt((x1 * x1) + (y1 * y1))  # 计算欧几里得距离作为基础奖励
        count = 1  # 计数器，用于控制奖励计算
        
        # 应用关节角度限制，确保动作在安全范围内
        self.future_state[11] = max(self.limit[11][0], min(self.limit[11][1], self.future_state[11]))
        self.future_state[13] = max(self.limit[13][0], min(self.limit[13][1], self.future_state[13]))
        self.future_state[15] = max(self.limit[15][0], min(self.limit[15][1], self.future_state[15]))
        
        # 执行多个仿真步
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        
        # 检查加速度计和陀螺仪数据是否在正常范围内
        for i in range(3):
            if self.acc_low[i] < acc[i] < self.acc_high[i] and self.gyro_low[i] < gyro[i] < self.gyro_high[i]:
                continue
            else:
                # 如果传感器数据异常，返回零奖励并结束回合
                print("传感器数据异常，返回零奖励并结束回合")
                reward = 0
                count = 0
                done = 1
                good = 0
                return self.next_state, reward, done, good, goal, count

        # 设置限制后的关节角度
        self.motors[11].setPosition(self.future_state[11])  # 设置左腿上部位置
        self.motors[13].setPosition(self.future_state[13])  # 设置左腿下部位置
        self.motors[15].setPosition(self.future_state[15])  # 设置左脚踝位置
        
        # 等待动作执行完成
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        self.robot.step(32)
        
        # 初始化返回值
        done = 0
        reward = -reward1  # 距离越小奖励越大
        good = 1
        
        # 检查是否发生碰撞
        for m in range(6):
            if self.touch_peng[m].getValue() == 1.0:
                # 如果发生碰撞，返回负奖励
                done = 0
                reward = -10.0
                good = 1
                count = 0
                return self.next_state, reward, done, good, goal, count



        # 检查脚部触摸传感器
        if self.touch1.getValue() == 1.0 or self.touch1_1.getValue() == 1.0 or self.touch1_2.getValue() == 1.0:
            # 如果脚部接触到物体（梯子），打印传感器值
            print("___________")
            print(self.touch1.getValue())
            print(self.touch1_1.getValue())
            print(self.touch1_2.getValue())
            
            # 等待一段时间
            timer = 0
            while self.robot.step(32) != -1:
                timer += 32
                if timer >= 2000:
                    break
                    
            # 获取触摸传感器值
            for j in range(len(self.touch)):
                self.touch_value[j] = self.touch[j].getValue()
                
            # 根据距离和触摸状态决定奖励和回合结束条件
            if (self.touch_value[0] == 1 or self.touch_value[1] == 1 or self.touch_value[2] == 1):
                reward = 50
                goal = 1
                count = 1
                done = 1
                good = 1
                return self.next_state, reward, done, good, goal, count
            else:
                reward = -2.0
                goal = 0
                count = 1
                done = 1
                good = 1

        # 获取当前关节角度并检查是否达到目标位置
        # for i in range(20):
        #     self.next_state[i] = self.motors_sensors[i].getValue()  # 获取当前关节角度
        #     self.cha_zhi = self.next_state[i] - self.future_state[i]  # 计算与目标角度的差值
        #     if -0.01 < self.cha_zhi < 0.01:
        #         # 如果差值很小，认为已达到目标位置
        #         continue
        #     else:
        #         # 如果差值较大，认为未达到目标位置
        #         count = 1
        #         reward = 0
        #         done = 0
        #         goal = 0
        #         good = 1
        #         break
                
        # 返回结果
        return self.next_state, reward, done, good, goal, count


class TaiStageRunner:
    """抬腿阶段封装：负责选动作、执行、奖励、经验、学习、保存和日志。"""

    def __init__(
        self,
        tai_agent,
        training_manager,
        log_writer_tai,
        log_file_latest_tai,
        tai_checkpoint_dir,
        max_steps=20,
        save_interval=400,
    ):
        self.tai_agent = tai_agent
        self.training_manager = training_manager
        self.log_writer_tai = log_writer_tai
        self.log_file_latest_tai = log_file_latest_tai
        self.tai_checkpoint_dir = tai_checkpoint_dir
        self.max_steps = max_steps
        self.save_interval = save_interval

    def _initial_last_actions(self, env):
        robot_state_initial = env.get_robot_state()
        return (
            robot_state_initial[12] if len(robot_state_initial) > 12 else 0.0,
            robot_state_initial[13] if len(robot_state_initial) > 13 else 0.0,
            robot_state_initial[14] if len(robot_state_initial) > 14 else 0.0,
        )

    def _map_gate_actions(
        self,
        discrete_upper,
        discrete_lower,
        discrete_ankle,
        action_leg_upper,
        action_leg_lower,
        action_ankle,
        last_action_leg_upper,
        last_action_leg_lower,
        last_action_ankle,
    ):
        action_leg_upper_exec = last_action_leg_upper if discrete_upper == 0 else action_leg_upper
        action_leg_lower_exec = last_action_leg_lower if discrete_lower == 0 else action_leg_lower
        action_ankle_exec = last_action_ankle if discrete_ankle == 0 else action_ankle
        return action_leg_upper_exec, action_leg_lower_exec, action_ankle_exec

    def _compute_reward(
        self,
        tai_episode,
        reward_env,
        count,
        gps_values,
        prev_distance,
        prev_foot_height,
        action_leg_upper,
        action_leg_lower,
        action_ankle,
        discrete_upper,
        discrete_lower,
        discrete_ankle,
        gate_activation,
    ):
        reward = 0.0
        if count == 1:
            x1 = gps_goal1[0] - gps_values[4][1]
            y1 = gps_goal1[1] - gps_values[4][2]
            distance = math.sqrt(x1 * x1 + y1 * y1)
            if prev_distance is not None:
                reward += (prev_distance - distance) * 10.0
            else:
                reward -= distance
            prev_distance = distance

            is_position_correct = distance <= 0.06
            if is_position_correct:
                reward += 5

            foot_height = float(gps_values[4][0])
            if prev_foot_height is not None:
                height_diff = foot_height - prev_foot_height
                if tai_episode < 5:
                    if height_diff > 0:
                        reward += height_diff * 5.0
                else:
                    if is_position_correct and height_diff < 0:
                        reward += -height_diff * 20.0
                        print("位置正确，鼓励踩踏！")
                    elif height_diff < 0:
                        reward += -height_diff * 3.0
                    else:
                        reward -= height_diff * 5.0
            prev_foot_height = foot_height
            reward += float(reward_env)
        else:
            prev_distance = None
            prev_foot_height = None
            reward = float(reward_env)

        reward -= 0.02 * (abs(action_leg_upper) + abs(action_leg_lower) + abs(action_ankle))

        if discrete_upper == 0 and discrete_lower == 0 and discrete_ankle == 0:
            gate_activation["all_off"] += 1
            reward -= 1.0
            print("警告：所有下半身离散动作都为0，给予惩罚-1.0")

        return reward, prev_distance, prev_foot_height

    def _save_model_if_needed(self, total_episode, tai_episode, done):
        if tai_episode % self.save_interval != 0 or done != 1:
            return

        save_path = os.path.join(self.tai_checkpoint_dir, f"tai_agent_{total_episode}_{tai_episode}.ckpt")
        print(f"保存模型到: {save_path}")
        checkpoint = {
            "episode": tai_episode,
            "policy_tai": self.tai_agent.policy.state_dict(),
            "optimizer_tai": self.tai_agent.optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)

    def _learn_if_ready(self, tai_episode, return_all, goal, gate_activation):
        if self.training_manager is not None:
            self.training_manager.increment_tai()
            if self.training_manager.should_learn_tai():
                loss_discrete, loss_continuous = self.tai_agent.learn()
                print("=" * 60)
                print(f"【抬腿模型学习】{self.training_manager.get_status()}")
            else:
                print(f"【抬腿模型累积经验】{self.training_manager.get_status()}")
                return 0, 0
        else:
            loss_discrete, loss_continuous = self.tai_agent.learn()
            print("=" * 60)

        print(f"【第 {tai_episode} 回合训练完成】")
        print(f"  累积奖励 (return_all): {return_all:.4f}")
        print(f"  目标达成 (goal): {goal}")
        print(f"  离散损失 (loss_discrete): {loss_discrete:.6f}")
        print(f"  连续损失 (loss_continuous): {loss_continuous:.6f}")
        print(f"  总损失 (total loss): {loss_discrete + loss_continuous:.6f}")
        self._print_gate_activation(gate_activation)
        print("=" * 60)
        return loss_discrete, loss_continuous

    def _print_gate_activation(self, gate_activation):
        if gate_activation["steps"] <= 0:
            return

        print("【门控激活率】")
        print(f"  LegUpper 激活率: {gate_activation['upper'] / gate_activation['steps']:.2%}")
        print(f"  LegLower 激活率: {gate_activation['lower'] / gate_activation['steps']:.2%}")
        print(f"  Ankle 激活率: {gate_activation['ankle'] / gate_activation['steps']:.2%}")
        print(f"  全关闭次数: {gate_activation['all_off']} / {int(gate_activation['steps'])}")

    def _log_episode(self, tai_episode, total_episode, loss_discrete, loss_continuous, return_all, steps, goal, gate_activation):
        tai_success = bool(goal == 1)
        self.log_writer_tai.add_episode(
            episode_num=tai_episode,
            total_episode=total_episode,
            loss_discrete=loss_discrete,
            loss_continuous=loss_continuous,
            episode_reward=return_all,
            episode_steps=steps,
            tai_success=tai_success,
            goal=goal,
            gate_upper_ratio=gate_activation["upper"] / gate_activation["steps"] if gate_activation["steps"] > 0 else 0,
            gate_lower_ratio=gate_activation["lower"] / gate_activation["steps"] if gate_activation["steps"] > 0 else 0,
            gate_ankle_ratio=gate_activation["ankle"] / gate_activation["steps"] if gate_activation["steps"] > 0 else 0,
        )
        self.log_writer_tai.save(self.log_file_latest_tai)

    def _reset_after_episode(self, env):
        print("抬腿回合结束，重置环境...")
        env.darwin._set_left_leg_initpose()
        env.darwin.robot_reset()
        env.reset()
        print("等待稳定...")
        for _ in range(40):
            env.robot.step(env.timestep)
        print("等待一秒...")
        env.wait(1000)

    def run_episode(self, env, total_episode, tai_episode, catch_success):
        if not catch_success:
            print("未检测到抓取成功，跳过抬腿阶段。")
            return False

        print("开始抬腿！")
        env.darwin.tai_leg_L1()
        env.darwin.tai_leg_L2()

        return_all = 0
        prev_distance = None
        prev_foot_height = None
        goal = 0
        done = 0
        steps = 0
        catch_flag = 0.0
        gate_activation = {"upper": 0.0, "lower": 0.0, "ankle": 0.0, "all_off": 0, "steps": 0}
        last_action_leg_upper, last_action_leg_lower, last_action_ankle = self._initial_last_actions(env)

        print("____________________")

        while True:
            obs_img, obs_tensor = env.get_img(steps)
            robot_state = validate_and_clean_data(env.get_robot_state())

            tai_dict = self.tai_agent.choose_action(
                episode_num=tai_episode,
                obs=[obs_img, robot_state],
                x_graph=robot_state,
            )

            tai_discrete_action = tai_dict['discrete_action']
            tai_continuous_action = tai_dict['continuous_action']
            tai_continuous_log_prob = tai_dict['continuous_log_prob']
            tai_value = tai_dict['value']

            action_leg_upper = float(tai_continuous_action[0])
            action_leg_lower = float(tai_continuous_action[1])
            action_ankle = float(tai_continuous_action[2])
            discrete_upper = int(tai_discrete_action[0])
            discrete_lower = int(tai_discrete_action[1])
            discrete_ankle = int(tai_discrete_action[2])

            action_leg_upper_exec, action_leg_lower_exec, action_ankle_exec = self._map_gate_actions(
                discrete_upper,
                discrete_lower,
                discrete_ankle,
                action_leg_upper,
                action_leg_lower,
                action_ankle,
                last_action_leg_upper,
                last_action_leg_lower,
                last_action_ankle,
            )
            last_action_leg_upper = action_leg_upper_exec
            last_action_leg_lower = action_leg_lower_exec
            last_action_ankle = action_ankle_exec

            gate_activation["steps"] += 1
            gate_activation["upper"] += discrete_upper
            gate_activation["lower"] += discrete_lower
            gate_activation["ankle"] += discrete_ankle

            print("第", steps + 1, "步")
            print(f"【tai_agent门控控制】离散动作: [{discrete_upper}, {discrete_lower}, {discrete_ankle}]")
            print(f"【原始连续动作】LegUpper: {action_leg_upper:.4f}, LegLower: {action_leg_lower:.4f}, Ankle: {action_ankle:.4f}")
            print(f"【最终执行动作】LegUpper: {action_leg_upper_exec:.4f}, LegLower: {action_leg_lower_exec:.4f}, Ankle: {action_ankle_exec:.4f}")

            gps_values = validate_and_clean_data(env.print_gps())
            next_state, reward_env, done, good, goal, count = env.step2(
                robot_state,
                action_leg_upper_exec,
                action_leg_lower_exec,
                action_ankle_exec,
                steps,
                catch_flag,
                gps_values[4],
                gps_values[0],
                gps_values[1],
                gps_values[2],
                gps_values[3],
            )

            reward, prev_distance, prev_foot_height = self._compute_reward(
                tai_episode,
                reward_env,
                count,
                gps_values,
                prev_distance,
                prev_foot_height,
                action_leg_upper,
                action_leg_lower,
                action_ankle,
                discrete_upper,
                discrete_lower,
                discrete_ankle,
                gate_activation,
            )
            return_all += reward
            steps += 1

            next_obs_img, next_obs_tensor = env.get_img(steps)
            should_store = not (done == 1 and steps <= 2 and good != 1 and reward == 0)
            if should_store:
                self.tai_agent.store_transition(
                    state=[obs_img, robot_state, robot_state],
                    discrete_action=tai_discrete_action,
                    continuous_action=tai_continuous_action,
                    reward=reward,
                    next_state=[next_obs_img, robot_state, next_state],
                    done=done,
                    value=tai_value,
                    discrete_log_prob=tai_dict['discrete_log_prob'],
                    continuous_log_prob=tai_continuous_log_prob,
                )
                print(f"  已存储经验 reward={reward:.4f}")
            else:
                print(f"  跳过无效样本：done={done}, steps={steps}, good={good}")

            obs_tensor = next_obs_tensor
            robot_state = env.get_robot_state()

            if steps >= self.max_steps:
                done = 1

            self._save_model_if_needed(total_episode, tai_episode, done)

            if tai_episode > 0 and done == 1:
                loss_discrete, loss_continuous = self._learn_if_ready(tai_episode, return_all, goal, gate_activation)
                self._log_episode(
                    tai_episode,
                    total_episode,
                    loss_discrete,
                    loss_continuous,
                    return_all,
                    steps,
                    goal,
                    gate_activation,
                )

            print("done:", done)

            if done == 1 or steps > self.max_steps:
                self._reset_after_episode(env)
                break

        return True