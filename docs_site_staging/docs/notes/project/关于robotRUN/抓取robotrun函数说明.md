# 抓取 RobotRun 函数说明

## 1. 文件定位与职责
- 代码位置：`python_scripts/PPO/RobotRun1.py`
- 核心类：`RobotRun(Darwin)`
- 作用：在抓取阶段执行单步控制逻辑，完成动作映射、约束检查、抓取判定、奖励计算，并返回 PPO 训练所需四元组：`next_state, reward, done, catch_success`。

---

## 2. 输入输出总览

### 2.1 构造输入
`RobotRun(robot, discrete_action, continuous_action, step)`
- `robot`：Webots 机器人实例
- `discrete_action`：离散门控动作（通常长度 2）
  - 第 1 维：肩部是否更新
  - 第 2 维：手臂是否更新
- `continuous_action`：连续动作（通常长度 2）
  - 第 1 维：肩部控制值
  - 第 2 维：手臂控制值
- `step`：当前环境步数

### 2.2 run 输出
`run()` 返回：
1. `next_state`：下一时刻关节状态（长度 20）
2. `reward`：本步奖励
3. `done`：是否终止当前 episode（0/1）
4. `catch_success`：是否“抓取成功且目标正确”（bool）

---

## 3. 核心状态字段
- `catch_flag`
  - `0.0`：执行动作阶段
  - `1.0`：已触碰，进入闭合验证阶段
  - 当前实现中由触碰传感器动态设置（不再依赖固定步数）
- `return_flag_list`
  - `reward`
  - `done`
  - `catch_success`
  - `catch_state`：抓取状态标签
  - `stair_score_1`
  - `stair_score_2`
  - `is_target_stair`
  - `distance`

---

## 4. 函数级说明

## `__init__(robot, discrete_action, continuous_action, step)`
- 功能：初始化单步执行器，准备动作、目标关节、传感器引用和返回字段。
- 关键处理：
  - 读取机器人当前关节状态和 GPS
  - 调用 `_map_policy_actions` 完成动作门控
  - 计算目标位姿 `self.next`
  - 生成未来状态 `self.future_state`
  - 初始化触碰/碰撞传感器引用
  - 初始化 `return_flag_list`

## `_to_float(value, default=0.0)`
- 功能：将输入安全转成 `float`。
- 目的：兼容 Python 标量、Tensor、Numpy 标量等，避免类型错误。

## `_map_policy_actions(discrete_action, continuous_action)`
- 功能：根据离散动作决定连续动作是否生效。
- 逻辑：
  - 离散为 0：沿用上一步动作
  - 离散为 1：使用当前策略输出
- 返回：`(mapped_shoulder, mapped_arm)`
- 附带：更新类级缓存 `_prev_shoulder_action`、`_prev_arm_action`

## `_compute_distance_metrics()`
- 功能：计算目标相关几何指标。
- 输出：`stair_score_1, stair_score_2, distance, is_target_stair`
- 说明：
  - `stair_score_1/2`：两路台阶匹配评分
  - `distance`：机器人与抓取目标的距离
  - `is_target_stair`：目标台阶命中标记（评分和阈值判定）

## `_check_joint_limits()`
- 功能：检查 `future_state` 是否越过 `Darwin_config.limit` 关节边界。
- 返回：`True/False`

## `_check_imu_limits()`
- 功能：检查加速度计和陀螺仪是否处于安全范围。
- 返回：`True/False`

## `_apply_upper_body_action()`
- 功能：将计算出的目标位置写入上肢关节电机。
- 控制关节：左右肩、左右臂（索引 1/0/5/4）

## `_wait_action_settle(max_steps=100, tol=0.01)`
- 功能：等待关键关节到位，防止动作尚未稳定就做抓取判定。
- 返回：
  - `True`：在给定步数内收敛
  - `False`：超时或仿真结束

## `_has_any_grasp_touch()`
- 功能：检测抓爪任意触碰传感器是否触发。
- 返回：`True/False`

## `_close_grasp_and_read_pair(wait_ms=2000)`
- 功能：闭合抓爪，并按旧模板读取两路触碰结果。
- 返回：`(success, failed)`
  - `success=1`：与成功触碰模板匹配
  - `failed=1`：与失败触碰模板匹配

## `_check_collision_constraint()`
- 功能：检查手臂/腿部碰撞触发传感器。
- 返回：`True/False`（True 表示发生碰撞）

## `_refresh_next_state()`
- 功能：刷新 `next_state`，供 PPO 作为下一时刻输入。

## `_check_tracking_constraint(tol=0.005)`
- 功能：检查执行后实测关节与 `future_state` 偏差是否超阈值。
- 返回：`True/False`

## `_log_constraint_trigger(name, details="")`
- 功能：统一打印约束触发日志，便于调试。

## `compute_reward(distance, prev_distance, done, success, failed, goal, collision, imu_ok, joints_ok)`
- 功能：统一计算本步奖励。
- 奖励项组成：
  - 距离变化奖励（朝目标靠近）
  - 接近奖励（近距离正激励）
  - 动作惩罚（过小动作、过大动作）
  - 时间惩罚（步数越大惩罚越高）
  - 碰撞惩罚
  - 终局奖惩（抓对/抓错/失败）

## `run()`
- 功能：单步主流程，串联约束、动作、抓取、奖励与返回。
- 主流程：
1. 推进一次仿真并初始化返回字段
2. 计算距离指标，检查关节/IMU 约束
3. 若约束触发，`done=1` 终止
4. `catch_flag=0`：执行上肢动作并等待稳定
5. 碰撞检查、触碰触发闭合、抓取结果判定
6. `catch_flag=1` 分支：直接做闭合验证
7. 计算奖励并刷新 `next_state`
8. 返回 `(next_state, reward, done, catch_success)`

---

## 5. catch_state 状态定义
- `continue_search`：未触碰，继续搜索/靠近
- `attempt_now`：已触碰，开始闭合尝试
- `attempt_failed`：闭合失败
- `grasped_wrong`：抓到但目标错误
- `grasped_right`：抓到且目标正确
- `terminated_by_constraint`：被约束中止（关节/IMU/碰撞/跟踪）

---

## 6. 与 PPO 的关系
- `done`：用于截断轨迹，影响 GAE 与回报传播。
- `reward`：作为策略优化的直接学习信号。
- `catch_success`：用于阶段切换（如是否进入抬腿阶段）。

---

## 7. 调试建议
- 已有 step 开始与结束打印，可快速观察：
  - 动作输入
  - 距离变化
  - catch_state 变化
  - reward/done/catch_success
- 若训练不稳定，优先检查：
  - `touch_T / touch_F` 模板是否与传感器真实输出一致
  - `tracking tol` 是否过严
  - 终局奖励比例是否过大导致策略抖动
