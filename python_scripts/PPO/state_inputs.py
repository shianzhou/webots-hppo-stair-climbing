"""State builders for decision, catch and tai PPO stages."""
import math

import numpy as np

from python_scripts.Project_config import gps_goal1


TAI_STATE_DIM = 40
CATCH_STATE_DIM = 20
DECISION_STATE_DIM = 6


def validate_and_clean_data(data, default_value=0.0):
    """Validate sensor data and replace NaN/Inf values before PPO uses it."""
    if isinstance(data, (list, tuple)):
        return [validate_and_clean_data(x, default_value) for x in data]
    if isinstance(data, np.ndarray):
        return np.nan_to_num(data, nan=default_value, posinf=default_value, neginf=-default_value)
    if isinstance(data, (int, float, np.integer, np.floating)):
        value = float(data)
        if np.isnan(value) or np.isinf(value):
            return default_value
        return value
    return data


def _fixed_state(values, feature_dim, default_value=0.0):
    state = np.full(int(feature_dim), float(default_value), dtype=np.float32)
    values = np.asarray(validate_and_clean_data(values), dtype=np.float32).reshape(-1)
    count = min(state.shape[0], values.shape[0])
    if count > 0:
        state[:count] = values[:count]
    return np.nan_to_num(state, nan=default_value, posinf=default_value, neginf=-default_value)


def _safe_touch(env, name):
    try:
        return float(env.darwin.get_touch_sensor_value(name))
    except Exception:
        try:
            return float(env.get_touch_sensor_value(name))
        except Exception:
            return 0.0


def _safe_gps_values(env):
    try:
        return validate_and_clean_data(env.print_gps())
    except Exception:
        return []


def _safe_imu(env):
    acc_feat = [0.0, 0.0, 0.0]
    gyro_feat = [0.0, 0.0, 0.0]
    try:
        acc = env.darwin.accelerometer.getValues()
        acc_feat = [
            (float(acc[0]) - 520.0) / 100.0,
            (float(acc[1]) - 500.0) / 100.0,
            (float(acc[2]) - 640.0) / 120.0,
        ]
    except Exception:
        pass
    try:
        gyro = env.darwin.gyro.getValues()
        gyro_feat = [
            (float(gyro[0]) - 510.0) / 20.0,
            (float(gyro[1]) - 510.0) / 20.0,
            (float(gyro[2]) - 510.0) / 20.0,
        ]
    except Exception:
        pass
    return validate_and_clean_data(acc_feat), validate_and_clean_data(gyro_feat)


def build_decision_state(env, feature_dim=DECISION_STATE_DIM):
    """Read hand pressure sensors as the upper decision state's fixed input."""
    pressure_values = np.array([
        _safe_touch(env, 'grasp_L1'),
        _safe_touch(env, 'grasp_L1_1'),
        _safe_touch(env, 'grasp_L1_2'),
        _safe_touch(env, 'grasp_R1'),
        _safe_touch(env, 'grasp_R1_1'),
        _safe_touch(env, 'grasp_R1_2'),
    ], dtype=np.float32)

    state = _fixed_state(pressure_values, feature_dim)
    pressure_detected = bool(np.any(pressure_values > 0.0))
    return state, pressure_values, pressure_detected


def build_catch_state(env, step, feature_dim=CATCH_STATE_DIM):
    """Build the catch stage observation while preserving the existing 20D state."""
    obs_img, obs_tensor = env.get_img(step)
    robot_state = _fixed_state(env.get_robot_state(), feature_dim).tolist()
    obs = (obs_tensor, robot_state)
    x_graph = robot_state
    return obs_img, obs_tensor, robot_state, obs, x_graph


def build_tai_state(
    env,
    robot_state,
    prev_robot_state,
    prev_exec_actions,
    prev_discrete_actions,
    feature_dim=TAI_STATE_DIM,
):
    """Build a tai/stepping state with task-specific proprioception and contact cues."""
    rs = np.asarray(validate_and_clean_data(robot_state), dtype=np.float32).reshape(-1)
    if prev_robot_state is None:
        prs = rs.copy()
    else:
        prs = np.asarray(validate_and_clean_data(prev_robot_state), dtype=np.float32).reshape(-1)
        if prs.shape[0] != rs.shape[0]:
            prs = rs.copy()

    def idx(i):
        return float(rs[i]) if 0 <= i < rs.shape[0] else 0.0

    def didx(i):
        if 0 <= i < rs.shape[0] and 0 <= i < prs.shape[0]:
            return float(rs[i] - prs[i])
        return 0.0

    gps_values = _safe_gps_values(env)
    foot_gps = gps_values[4] if len(gps_values) > 4 else [0.0, 0.0, 0.0]
    foot_x = float(foot_gps[1]) if len(foot_gps) > 1 else 0.0
    foot_y = float(foot_gps[2]) if len(foot_gps) > 2 else 0.0

    if isinstance(gps_goal1, (list, tuple)) and len(gps_goal1) >= 2:
        goal_x = float(gps_goal1[0])
        goal_y = float(gps_goal1[1])
    else:
        goal_x, goal_y = 0.058, 0.023

    dx = goal_x - foot_x
    dy = goal_y - foot_y
    dist = math.sqrt(dx * dx + dy * dy)

    acc_feat, gyro_feat = _safe_imu(env)
    foot_touches = [
        _safe_touch(env, 'foot_L1'),
        _safe_touch(env, 'foot_L2'),
        _safe_touch(env, 'foot_L3'),
    ]

    prev_exec = _fixed_state(prev_exec_actions, 3)
    prev_disc = _fixed_state(prev_discrete_actions, 3)

    feature = [
        idx(10), idx(11), idx(12), idx(13), idx(14), idx(15),
        didx(10), didx(11), didx(12), didx(13), didx(14), didx(15),
        foot_x, foot_y, dx, dy, dist,
        *acc_feat,
        *gyro_feat,
        *foot_touches,
        float(prev_exec[0]), float(prev_exec[1]), float(prev_exec[2]),
        float(prev_disc[0]), float(prev_disc[1]), float(prev_disc[2]),
    ]
    return _fixed_state(feature, feature_dim)
