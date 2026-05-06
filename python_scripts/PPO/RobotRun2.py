"""RobotRun2 controller for the tai/stepping stage."""
import math
from python_scripts.Webots_interfaces import Darwin
from python_scripts.Project_config import Darwin_config
from python_scripts.Project_config import gps_goal1


class RobotRun2(Darwin):
    """Execute one lower-body stepping action and return environment feedback."""

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
        self.robot = robot
        self.timestep = 32
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

        self.success_dist_thresh = 0.04
        self.miss_penalty_base = 4.0
        self.miss_penalty_scale = 12.0
        self.action_settle_timeout = 100
        self.action_settle_tolerance = 0.01

        self.motors = []
        self.motors_sensors = []
        self.touch_value = [0.0, 0.0, 0.0]
        self.next_state = [0.0 for _ in range(20)]
        self.future_state = [value for value in self.robot_state]
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
            self.robot_state[11] + self.action_leg_upper,
            self.robot_state[13] + self.action_leg_lower,
            self.robot_state[15] + self.action_ankle,
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
        for target_idx, joint_idx in enumerate(self.leg_joint_indices):
            raw_value = self.future_state[joint_idx]
            clamped_value = self._clamp_joint(joint_idx, raw_value)
            self.future_state[joint_idx] = clamped_value
            clamped_targets.append(clamped_value)
            if raw_value != clamped_value:
                print(f"[CLAMP] step={self.step} {joint_idx}:{raw_value:.3f}->{clamped_value:.3f}")
        return clamped_targets

    def _check_imu(self):
        acc = self.accelerometer.getValues()
        gyro = self.gyro.getValues()
        for i in range(3):
            if not (self.acc_low[i] < acc[i] < self.acc_high[i] and self.gyro_low[i] < gyro[i] < self.gyro_high[i]):
                print("传感器数据异常，返回零奖励并结束回合")
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

    def _read_left_foot_touch(self, wait_ms=2000):
        if not any(sensor.getValue() > 0.0 for sensor in self.touch):
            return False

        print("___________")
        for sensor in self.touch:
            print(sensor.getValue())

        timer = 0
        while self.robot.step(self.timestep) != -1:
            timer += self.timestep
            if timer >= wait_ms:
                break

        for i, sensor in enumerate(self.touch):
            self.touch_value[i] = sensor.getValue()
        return any(value > 0.0 for value in self.touch_value)

    def _compute_step_touch_reward(self, distance):
        if distance < self.success_dist_thresh:
            closeness_ratio = max(0.0, (self.success_dist_thresh - distance) / self.success_dist_thresh)
            reward = 50.0 + 80.0 * closeness_ratio
            print(
                f"[STEP_OK] step={self.step} dist={distance:.4f} < {self.success_dist_thresh:.4f}, "
                f"closeness={closeness_ratio:.3f}, reward={reward:.2f}"
            )
            return reward, 1

        miss_ratio = min(1.0, max(0.0, (distance - self.success_dist_thresh) / self.success_dist_thresh))
        miss_penalty = self.miss_penalty_base + self.miss_penalty_scale * miss_ratio
        print(
            f"[STEP_OFF_TARGET] step={self.step} dist={distance:.4f} >= {self.success_dist_thresh:.4f}, "
            f"miss_ratio={miss_ratio:.3f}, penalty={miss_penalty:.2f}"
        )
        return -miss_penalty, 0

    def _refresh_next_state(self):
        for i in range(min(20, len(self.motors_sensors))):
            self.next_state[i] = self.motors_sensors[i].getValue()

    def _finish(self, reward, done, good, goal, count):
        self._refresh_next_state()
        return self.next_state, reward, done, good, goal, count

    def run(self):
        self.robot.step(self.timestep)

        target = self._apply_joint_limits()
        if not self._check_imu():
            return self._finish(reward=0.0, done=1, good=0, goal=0, count=0)

        self._set_leg_targets(target)
        reached, used_frames = self._wait_leg_target_reached(target)
        print(
            f"[SETTLE] step={self.step} reached={int(reached)} "
            f"frames={used_frames}/{self.action_settle_timeout} tol={self.action_settle_tolerance:.3f}"
        )

        total_distance = self._read_step_distance_after_action()
        reward = -total_distance
        done = 0
        good = 1
        goal = 0
        count = 1

        if self._has_collision():
            return self._finish(reward=-10.0, done=0, good=1, goal=0, count=0)

        if self._read_left_foot_touch():
            reward, goal = self._compute_step_touch_reward(total_distance)
            return self._finish(reward=reward, done=1, good=1, goal=goal, count=1)

        return self._finish(reward=reward, done=done, good=good, goal=goal, count=count)
