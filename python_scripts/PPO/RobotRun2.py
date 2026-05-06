"""RobotRun2 controller for the tai/stepping stage."""
import math

from python_scripts.Project_config import gps_goal1
from python_scripts.Webots_interfaces import Darwin


class RobotRun(Darwin):
    """Execute one lower-body stepping action and return environment feedback."""

    _prev_distance = None
    _prev_center_error = None
    _prev_foot_height = None

    motor_names = (
        'ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
        'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL', 'PelvR',
        'PelvL', 'LegUpperR', 'LegUpperL', 'LegLowerR', 'LegLowerL',
        'AnkleR', 'AnkleL', 'FootR', 'FootL', 'Neck', 'Head', 'GraspL', 'GraspR'
    )
    leg_joint_indices = (11, 13, 15)
    limits = [
        [-3.14, 3.14], [-3.14, 2.85], [-0.68, 2.3], [-2.25, 0.77],
        [-1.65, 1.16], [-1.18, 1.63], [-2.42, 0.66], [-0.69, 2.5],
        [-1.01, 1.01], [-1, 0.93], [-1.77, 0.45], [-0.5, 1.68],
        [-0.02, 2.25], [-2.25, 0.03], [-1.24, 1.38], [-1.39, 1.22],
        [-0.68, 1.04], [-1.02, 0.6], [-1.81, 1.81], [-0.36, 0.94],
    ]
    acc_low = [480, 450, 580]
    acc_high = [560, 530, 700]
    gyro_low = [500, 500, 500]
    gyro_high = [520, 520, 520]

    def _log_constraint_trigger(self, name, details=""):
        msg = f"[Constraint Triggered] step={self.step} | {name}"
        if details:
            msg += f" | {details}"
        print(msg)

    def __init__(
        self,
        robot,
        state,
        action_leg_upper,
        action_leg_lower,
        action_ankle,
        step,
        zhua,
        gps0,
        gps1,
        gps2,
        gps3,
        gps4,
    ):
        super().__init__(robot)
        self.step = step
        self.goal = gps_goal1
        self.robot_state = state
        self.gps0 = gps0
        self.gps1 = gps1
        self.gps2 = gps2
        self.gps3 = gps3
        self.gps4 = gps4
        self.if_jia = zhua

        self.action_leg_upper = float(action_leg_upper)
        self.action_leg_lower = float(action_leg_lower)
        self.action_ankle = float(action_ankle)

        self.leg_upper_delta = 1.09 * self.action_leg_upper + 0.59
        self.leg_lower_delta = 1.14 * self.action_leg_lower - 1.11
        self.ankle_delta = 1.305 * self.action_ankle - 0.085

        self.success_dist_thresh = 0.04
        self.distance_reward_scale = 15.0
        self.proximity_reward_scale = 5.0
        self.progress_reward_scale = 80.0
        self.center_progress_reward_scale = 120.0
        self.lateral_penalty_scale = 4.0
        self.action_penalty_scale = 0.02
        self.step_penalty_scale = 0.012
        self.collision_penalty = -8.0
        self.success_reward = 400.0
        self.success_bonus_scale = 120.0
        self.wrong_touch_penalty = -2.0
        self.failed_touch_penalty = -5.0
        self.miss_penalty_base = 4.0
        self.miss_penalty_scale = 12.0
        self.far_penalty = 3.0
        self.notouch_penalty = -10.0

        self.action_settle_timeout = 100
        self.action_settle_tolerance = 0.01

        self.motors = []
        self.motors_sensors = []
        self.touch_value = [0.0, 0.0, 0.0]
        self.future_state = [value for value in self.robot_state]
        self.next_state = [0.0 for _ in range(20)]
        self.next = self._compute_targets()

        self._init_devices()

    def _init_devices(self):
        for motor_name in self.motor_names:
            motor = self.robot.getDevice(motor_name)
            sensor = self.robot.getDevice(motor_name + 'S')
            sensor.enable(self.timestep)
            self.motors.append(motor)
            self.motors_sensors.append(sensor)

        self.accelerometer = self.robot.getDevice('Accelerometer')
        self.gyro = self.robot.getDevice('Gyro')
        self.foot_gps1 = self.robot.getDevice('foot_gps1')
        self.foot_gps1.enable(self.timestep)

        self.touch = [
            self.robot.getDevice('touch_foot_L1'),
            self.robot.getDevice('touch_foot_L2'),
            self.robot.getDevice('touch_foot_L3'),
        ]
        self.touch_peng = [
            self.robot.getDevice('touch_arm_L1'),
            self.robot.getDevice('touch_arm_R1'),
            self.robot.getDevice('touch_leg_L1'),
            self.robot.getDevice('touch_leg_L2'),
            self.robot.getDevice('touch_leg_R1'),
            self.robot.getDevice('touch_leg_R2'),
        ]

        for sensor in self.touch + self.touch_peng:
            sensor.enable(self.timestep)

    def _compute_targets(self):
        targets = [
            self.robot_state[11] + self.leg_upper_delta,
            self.robot_state[13] + self.leg_lower_delta,
            self.robot_state[15] + self.ankle_delta,
        ]
        self.future_state[11] = targets[0]
        self.future_state[13] = targets[1]
        self.future_state[15] = targets[2]
        print("变化动作")
        print(targets)
        return targets

    def _clamp_joint(self, idx, value):
        low, high = self.limits[idx]
        return max(low, min(high, value))

    def _apply_joint_limits(self):
        clamped_targets = []
        for joint_idx in self.leg_joint_indices:
            raw_value = self.future_state[joint_idx]
            clamped_value = self._clamp_joint(joint_idx, raw_value)
            self.future_state[joint_idx] = clamped_value
            clamped_targets.append(clamped_value)
            if raw_value != clamped_value:
                print(f"[CLAMP] step={self.step} joint={joint_idx} {raw_value:.3f}->{clamped_value:.3f}")
        return clamped_targets

    def _check_joint_limits(self):
        for joint_idx in self.leg_joint_indices:
            low, high = self.limits[joint_idx]
            if not (low <= self.future_state[joint_idx] <= high):
                return False
        return True

    def _check_imu_limits(self):
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        acc_ok = all(self.acc_low[i] < acc[i] < self.acc_high[i] for i in range(3))
        gyro_ok = all(self.gyro_low[i] < gyro[i] < self.gyro_high[i] for i in range(3))
        print(f"  [IMU] acc={[f'{v:.1f}' for v in acc]} (ok={acc_ok}) | gyro={[f'{v:.1f}' for v in gyro]} (ok={gyro_ok})")
        for i in range(3):
            if not (
                self.acc_low[i] < acc[i] < self.acc_high[i]
                and self.gyro_low[i] < gyro[i] < self.gyro_high[i]
            ):
                print(
                    f"    ⚠️ IMU 超出范围 [i={i}] acc[{i}]={acc[i]} (range {self.acc_low[i]}-{self.acc_high[i]}) | "
                    f"gyro[{i}]={gyro[i]} (range {self.gyro_low[i]}-{self.gyro_high[i]})"
                )
                return False
        return True

    def _set_leg_targets(self, target):
        self.motors[11].setPosition(float(target[0]))
        self.motors[13].setPosition(float(target[1]))
        self.motors[15].setPosition(float(target[2]))

    def _wait_leg_target_reached(self, target, timeout=None, tolerance=None):
        timeout = self.action_settle_timeout if timeout is None else timeout
        tolerance = self.action_settle_tolerance if tolerance is None else tolerance

        for frame_idx in range(int(timeout)):
            if self.robot.step(self.timestep) == -1:
                return False, frame_idx + 1
            current_positions = [
                float(self.motors_sensors[11].getValue()),
                float(self.motors_sensors[13].getValue()),
                float(self.motors_sensors[15].getValue()),
            ]
            if all(abs(float(target[i]) - current_positions[i]) <= tolerance for i in range(3)):
                return True, frame_idx + 1

        return False, int(timeout)

    def _read_step_distance_after_action(self):
        foot_gps_now = self.foot_gps1.getValues()
        x1 = self.goal[0] - float(foot_gps_now[1])
        y1 = self.goal[1] - float(foot_gps_now[2])
        return math.sqrt(x1 ** 2 + y1 ** 2)

    def _has_collision(self):
        return any(sensor.getValue() > 0.0 for sensor in self.touch_peng)

    def _has_foot_contact(self):
        return any(sensor.getValue() > 0.0 for sensor in self.touch)

    def _read_left_foot_touch(self, wait_ms=2000):
        pre_vals = {
            'foot_L1': self.touch[0].getValue(),
            'foot_L2': self.touch[1].getValue(),
            'foot_L3': self.touch[2].getValue(),
        }
        print(f"  [踩踏前传感器] {pre_vals}")

        if not self._has_foot_contact():
            print("  [诊断] 未检测到脚底接触，跳过闭合/判定")
            return False

        timer = 0
        while self.robot.step(self.timestep) != -1:
            timer += self.timestep
            if timer >= wait_ms:
                break

        for i, sensor in enumerate(self.touch):
            self.touch_value[i] = sensor.getValue()
        post_vals = {
            'foot_L1': self.touch[0].getValue(),
            'foot_L2': self.touch[1].getValue(),
            'foot_L3': self.touch[2].getValue(),
        }
        print(f"  [踩踏后传感器] touch_value={self.touch_value} | all_foot={post_vals}")
        success = any(value > 0.0 for value in self.touch_value)
        failed = all(value == 0.0 for value in self.touch_value)
        print(f"  [传感器读取结果] touch_value={self.touch_value} | success={int(success)} | failed={int(failed)}")
        return success

    def _compute_reward(self, distance, foot_x, foot_y, done, goal, collision, touch_success):
        dx = float(self.goal[0]) - float(foot_x)
        dy = float(self.goal[1]) - float(foot_y)

        reward = -distance * self.distance_reward_scale
        if RobotRun._prev_distance is None:
            reward -= distance
        else:
            reward += (RobotRun._prev_distance - distance) * self.progress_reward_scale

        center_error = math.sqrt(dx ** 2 + (2.8 * dy) ** 2)
        reward -= 4.0 * center_error
        reward -= self.lateral_penalty_scale * abs(dy)
        if RobotRun._prev_center_error is not None:
            reward += (RobotRun._prev_center_error - center_error) * self.center_progress_reward_scale

        if distance > 0.25:
            reward -= self.far_penalty

        reward -= self.action_penalty_scale * (
            abs(self.action_leg_upper) + abs(self.action_leg_lower) + abs(self.action_ankle)
        )
        reward -= self.step_penalty_scale * float(self.step)

        if collision:
            reward += self.collision_penalty

        if done == 1:
            if touch_success:
                if goal == 1:
                    closeness_ratio = max(0.0, (self.success_dist_thresh - distance) / self.success_dist_thresh)
                    reward += self.success_reward + self.success_bonus_scale * closeness_ratio
                else:
                    miss_ratio = min(1.0, max(0.0, (distance - self.success_dist_thresh) / self.success_dist_thresh))
                    reward -= self.miss_penalty_base + self.miss_penalty_scale * miss_ratio
                    reward += self.wrong_touch_penalty
            else:
                reward += self.notouch_penalty

        return reward, center_error

    def _refresh_next_state(self):
        for i in range(min(20, len(self.motors_sensors))):
            self.next_state[i] = self.motors_sensors[i].getValue()

    def _finish(self, reward, done, good, goal, count):
        self._refresh_next_state()
        return self.next_state, reward, done, good, goal, count

    def run(self):
        self.robot.step(self.timestep)

        if int(self.step) <= 0:
            RobotRun._prev_distance = None
            RobotRun._prev_center_error = None
            RobotRun._prev_foot_height = None

        print(
            f"\n[Step {int(self.step)}] "
            f"leg_action={self.action_leg_upper:.3f},{self.action_leg_lower:.3f},{self.action_ankle:.3f} | "
            f"goal={self.goal}"
        )

        # 先将目标位夹紧到允许范围，避免“刚进入踩踏阶段就因目标越界直接结束”
        target = self._apply_joint_limits()
        print(f"  [关节目标] raw={self.next[0]:.3f},{self.next[1]:.3f},{self.next[2]:.3f} | clamped={target}")

        if not self._check_joint_limits():
            self._log_constraint_trigger('joint_limits', 'future_state still out of limits after clamp')
            self._refresh_next_state()
            return self._finish(reward=0.0, done=1, good=0, goal=0, count=0)

        if not self._check_imu_limits():
            self._log_constraint_trigger('imu_limits', 'accelerometer/gyro out of threshold')
            self._refresh_next_state()
            return self._finish(reward=0.0, done=1, good=0, goal=0, count=0)

        self._set_leg_targets(target)
        reached, used_frames = self._wait_leg_target_reached(target)
        print(
            f"[SETTLE] step={self.step} reached={int(reached)} "
            f"frames={used_frames}/{self.action_settle_timeout} tol={self.action_settle_tolerance:.3f}"
        )

        distance = self._read_step_distance_after_action()
        foot_gps_now = self.foot_gps1.getValues()
        foot_x = float(foot_gps_now[1])
        foot_y = float(foot_gps_now[2])
        goal = 0
        done = 0
        good = 1
        count = 1
        touch_success = False

        if self._has_collision():
            self._log_constraint_trigger('collision', 'arm/leg touch sensors are active')
            RobotRun._prev_distance = None
            RobotRun._prev_foot_height = None
            return self._finish(reward=self.collision_penalty, done=1, good=0, goal=0, count=0)

        if self._read_left_foot_touch():
            touch_success = True
            print(
                f"  [台阶匹配检查] distance={distance:.4f} | "
                f"success_thresh={self.success_dist_thresh:.4f}"
            )
            if distance <= self.success_dist_thresh:
                goal = 1
                self._log_constraint_trigger('target_check', 'grasped_right')
                print(f"[STEP_OK] step={self.step} dist={distance:.4f} < {self.success_dist_thresh:.4f}")
            else:
                self._log_constraint_trigger('target_check', 'grasped_wrong')
                print(f"[STEP_OFF_TARGET] step={self.step} dist={distance:.4f} >= {self.success_dist_thresh:.4f}")
            done = 1
        else:
            self._log_constraint_trigger('touch_result', 'close failed - no complete foot contact')

        reward, center_error = self._compute_reward(
            distance=distance,
            foot_x=foot_x,
            foot_y=foot_y,
            done=done,
            goal=goal,
            collision=False,
            touch_success=touch_success,
        )

        print(
            f"  [台阶奖励] dist={distance:.4f} | center_error={center_error:.4f} | "
            f"prev_dist={RobotRun._prev_distance} | prev_center_error={RobotRun._prev_center_error} | reward={reward:.4f}"
        )

        RobotRun._prev_distance = distance if done == 0 else None
        RobotRun._prev_center_error = center_error if done == 0 else None
        RobotRun._prev_foot_height = None if done == 1 else RobotRun._prev_foot_height

        return self._finish(reward=reward, done=done, good=good, goal=goal, count=count)


# 兼容旧调用，避免外部仍以RobotRun2引用时报错
RobotRun2 = RobotRun
