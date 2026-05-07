# 下载版踩踏逻辑总结

  

本文只总结下载版踩踏阶段逻辑，主要参考：

  

- `D:\downloads\PPO_episoid_2_1 (2).py`

- `D:\downloads\RobotRun2 (2).py`

- `D:\downloads\PPO_episoid_1 (1).py`

  

## 一句话结论

  

下载版 `PPO_tai_episoid()` 虽然日志里还叫“抬腿阶段”，但它训练的不是单纯把腿抬起来，而是“左脚踩踏/踩台阶子任务”：策略控制左腿三个关节，让左脚最终触碰目标台阶，并且动作后的左脚 GPS 落点足够接近 `gps_goal1`。

  

右脚复现、双脚站稳、进入台阶上抓取属于踩踏成功后的流程衔接，不属于 `PPO_tai_episoid()` 内部 PPO 训练核心。

  

## 入口与前置条件

  

下载版主流程在 `D:\downloads\PPO_episoid_1 (1).py` 中调用：

  

```python

tai_success = PPO_tai_episoid(

    existing_env=env,

    total_episode=total_episode,

    episode=tai_episoid,

    log_writer_tai=log_writer_tai,

    log_file_latest_tai=log_file_latest_tai,

    catch_success=catch_success,

    tai_agent=tai_agent,

    keep_pose_on_success=True,

    train_tai=train_tai,

)

```

  

`PPO_tai_episoid()` 的第一层判断是：

  

```python

if not catch_success:

    print("未检测到抓取成功，跳过抬腿阶段。")

    return False

```

  

所以下载版踩踏训练的前置条件是：抓取阶段已经成功，`catch_success=True`。如果抓取没成功，不会进入左脚踩踏训练。

  

进入后会先执行：

  

```python

env.darwin.tai_leg_L1()

env.darwin.tai_leg_L2()

```

  

这更像踩踏前的准备姿态。真正由 PPO 学习的动作，是后面循环中 `tai_agent.choose_action()` 输出的三关节控制。

  

## 状态输入

  

下载版为踩踏阶段单独构造了 `tai_state`，函数是：

  

```python

_build_tai_feature_state(...)

```

  

它不是只把 20 维关节状态原样传进去，而是拼出一组踩踏相关特征，主要包括：

  

| 输入内容 | 具体含义 |

| --- | --- |

| 双腿关键关节角 | `robot_state[10:15]` 附近的腿部关节，包含左右腿关键姿态 |

| 关节变化量 | 当前关节状态减去上一帧关节状态，近似表示角速度/变化趋势 |

| 左脚 GPS 与目标差值 | `foot_gps1` 的脚部位置、`gps_goal1 - foot_gps1`、距离 `dist` |

| IMU | 加速度计和陀螺仪特征，用于反映姿态稳定性 |

| 左脚触碰 | `foot_L1 / foot_L2 / foot_L3` 三个左脚触碰值 |

| 上一步执行动作 | 上一步实际执行的 `LegUpperL / LegLowerL / AnkleL` 动作 |

| 上一步离散门控 | 上一步三个关节的 discrete gate |

  

循环中每一步会读取图像和状态：

  

```python

_, obs_tensor = env.get_img(steps, imgs)

robot_state = env.get_robot_state()

tai_state = _build_tai_feature_state(...)

```

  

随后传给踩踏智能体：

  

```python

tai_dict = tai_agent.choose_action(

    episode_num=episode,

    obs=[obs_tensor, tai_state],

    x_graph=tai_state,

)

```

  

因此下载版踩踏策略的输入可以理解为：

  

```text

图像 obs_tensor + 踩踏特征 tai_state + 图/结构分支 x_graph=tai_state

```

  

其中 `tai_state` 是核心状态输入，里面已经包含脚到目标的距离、触碰、IMU、上一动作等踩踏闭环需要的信息。

  

## 动作输出

  

`tai_agent.choose_action()` 输出两类动作：

  

```python

tai_discrete_action = tai_dict['discrete_action']

tai_continuous_action = tai_dict['continuous_action']

```

  

它们都是三维，对应左腿三个关节：

  

| 维度 | 关节 | 连续动作含义 | 离散动作含义 |

| --- | --- | --- | --- |

| 0 | `LegUpperL` | 左大腿关节动作 | 是否采用本步新动作 |

| 1 | `LegLowerL` | 左小腿关节动作 | 是否采用本步新动作 |

| 2 | `AnkleL` | 左脚踝关节动作 | 是否采用本步新动作 |

  

离散动作是门控信号：

  

```python

if discrete_upper == 0:

    action_LegUpper_exec = last_action_LegUpper

else:

    action_LegUpper_exec = action_LegUpper

```

  

`LegLowerL` 和 `AnkleL` 同理。也就是说，策略不是每一步都必须更新三个关节；某个离散门控为 `0` 时，该关节保持上一时刻实际执行动作。

  

最终传入环境的是门控后的动作：

  

```python

env.step2(

    robot_state,

    action_LegUpper_exec,

    action_LegLower_exec,

    action_Ankle_exec,

    ...

)

```

  

## 三关节动作为什么不是任意值

  

从表面看，策略只输出三个连续动作，确实容易误解成“只是训练三个关节值”。但下载版的训练不是把这三个值本身当成功结果，而是把它们放进 Webots 环境里执行，再用脚部传感器和奖励筛选。

  

完整闭环是：

  

```text

tai_state/obs_tensor

  -> tai_agent 输出三维连续动作 + 三维离散门控

  -> 门控决定使用新动作还是上一动作

  -> env.step2() 进入 RobotRun2

  -> 动作转换为左腿三个关节目标角度

  -> 关节限位

  -> Webots 电机 setPosition()

  -> 等待关节真实到位

  -> 读取动作后的 foot_gps1 和左脚触碰

  -> 根据距离、触碰、偏踩、动作幅度、步数计算 reward

  -> PPO 用 reward 更新策略

```

  

所以三关节动作可以被探索，但不会被无条件认可。只有让左脚更接近目标、最终踩到 `gps_goal1` 附近的动作组合，才会得到更好的训练信号。无触碰、偏踩、距离过远、碰撞、姿态异常都会带来惩罚或终止。

  

## `target` 到底是什么

  

在下载版 `D:\downloads\RobotRun2 (2).py` 中，初始化时会根据当前机器人状态和动作计算：

  

```python

self.next = [

    self.robot_state[11] + self.LegUpper,

    self.robot_state[13] + self.LegLower,

    self.robot_state[15] + self.Ankle,

]

  

self.future_state[11] = self.next[0]

self.future_state[13] = self.next[1]

self.future_state[15] = self.next[2]

```

  

这里的 `target` 或 `final_target` 不是台阶坐标，而是本步要下发给 Webots 电机的三个左腿关节目标角度：

  

```text

target = 当前左腿关节角度 + 本步动作增量

```

  

对应关系是：

  

| `target` 下标 | 关节索引 | 关节名 |

| --- | --- | --- |

| `target[0]` | 11 | `LegUpperL` |

| `target[1]` | 13 | `LegLowerL` |

| `target[2]` | 15 | `AnkleL` |

  

台阶目标位置是 `gps_goal1`，不是 `target`。`target` 管的是“关节要转到哪里”，`gps_goal1` 管的是“脚最终应该踩到哪里”。

  

## 动作执行过程

  

下载版 `RobotRun2.run()` 中，动作执行的主要流程是：

  

1. 读取 IMU。

2. 将 `future_state[11] / [13] / [15]` 按关节限制裁剪。

3. 检查加速度计、陀螺仪是否在正常范围。

4. 将三个关节目标角度下发给 Webots 电机。

5. 等待电机实际到位。

6. 动作完成后读取左脚 GPS 和触碰传感器。

7. 返回 `next_state, reward, done, good, goal, count`。

  

下发动作的代码是：

  

```python

final_target = [

    float(self.future_state[11]),

    float(self.future_state[13]),

    float(self.future_state[15]),

]

self.motors[11].setPosition(final_target[0])

self.motors[13].setPosition(final_target[1])

self.motors[15].setPosition(final_target[2])

```

  

然后调用：

  

```python

reached, used_frames = self._wait_leg_target_reached(final_target)

```

  

`_wait_leg_target_reached()` 会循环推进 Webots 仿真，并读取三个关节的位置传感器：

  

```python

current_positions = [

    float(self.motors_sensors[11].getValue()),

    float(self.motors_sensors[13].getValue()),

    float(self.motors_sensors[15].getValue()),

]

```

  

只要三个关节当前位置都接近 `final_target`，误差小于 `action_settle_tolerance = 0.01`，就认为动作到位。最多等待 `action_settle_timeout = 100` 帧。

  

这个函数不是判断踩踏成功，也不是规划落脚动作。它只是确保刚刚 `setPosition()` 下发的三个关节目标已经尽量执行完成，然后再去读脚部 GPS 和触碰结果。

  

## 腿什么时候落下，怎么落下

  

下载版没有一个单独的“落腿函数”，也没有硬编码：

  

```text

先抬腿 -> 再前伸 -> 再落下

```

  

腿什么时候落下，取决于连续多个 step 中策略输出的三个关节目标角度。每一步策略只决定当前左大腿、左小腿、左脚踝该转到什么目标角度；Webots 物理仿真会根据这些目标角度驱动腿部运动。

  

如果多步动作形成了这样的效果：

  

```text

前几步：脚离开原位置，腿部姿态变化

中间步：脚向目标台阶方向靠近

后续步：脚底接触台阶

```

  

那么从外部看就是“从抬到落”的过程。这个过程不是代码显式写死的，而是由以下因素共同塑造：

  

- 策略每一步输出的三关节目标。

- 关节限位，避免目标角度越界。

- Webots 电机和物理仿真执行。

- 左脚触碰传感器检测接触。

- `foot_gps1` 到 `gps_goal1` 的距离奖励和成功奖励。

  

因此“落下”发生在脚底触碰传感器被触发时，但训练并不是直接命令“现在落下”，而是通过 reward 学习哪些关节组合更容易让脚落到目标台阶。

  

## 如何判断到目标位置

  

下载版有两层判断：环境层 `RobotRun2.run()` 和训练循环层 `PPO_tai_episoid()`。

  

环境层动作执行后读取：

  

```python

foot_gps_now = self.foot_gps1.getValues()

x1 = self.goal[0] - float(foot_gps_now[1])

y1 = self.goal[1] - float(foot_gps_now[2])

total_distance = math.sqrt(x1 ** 2 + y1 ** 2)

```

  

其中 `self.goal` 来自：

  

```python

self.goal = gps_goal1

```

  

也就是目标台阶位置。

  

随后检查左脚触碰：

  

```python

self.touch1.getValue()

self.touch1_1.getValue()

self.touch1_2.getValue()

```

  

它们对应：

  

```text

touch_foot_L1 / touch_foot_L2 / touch_foot_L3

```

  

成功条件是：

  

```python

left_touch_any = (

    self.touch_value[0] > 0.0

    or self.touch_value[1] > 0.0

    or self.touch_value[2] > 0.0

)

  

if left_touch_any:

    if total_distance < self.success_dist_thresh:

        goal = 1

        done = 1

```

  

下载版阈值是：

  

```python

self.success_dist_thresh = 0.04

```

  

训练循环层还有一个严格成功兜底：

  

```python

TAI_SUCCESS_DIST_THRESH = 0.04

strict_success = (goal == 1 and dist < TAI_SUCCESS_DIST_THRESH)

```

  

所以踩踏成功必须同时满足：

  

```text

左脚触碰台阶 + 动作后 foot_gps1 到 gps_goal1 的距离 < 0.04

```

  

只触碰但距离不够近，属于偏踩，不算成功。

  

## 奖励逻辑

  

下载版奖励主要在 `PPO_tai_episoid()` 循环中统一计算。核心目标是让脚越来越靠近目标台阶，并最终在正确位置触碰。

  

主要奖励/惩罚项包括：

  

| 项目 | 作用 |

| --- | --- |

| 基础距离惩罚 | `dist` 越大惩罚越大，鼓励脚靠近目标 |

| 进度奖励 | 如果比上一步更接近目标，则给正向进度奖励 |

| 中心误差惩罚 | 对横向偏差 `dy` 加更大权重，避免踩偏 |

| 成功奖励 | `goal == 1` 且 `dist < 0.04` 时给大额奖励 |

| 偏踩惩罚 | 脚碰到了但距离超过阈值，不算成功并惩罚 |

| 过远惩罚 | `dist > 0.25` 时额外惩罚 |

| 动作幅度惩罚 | 三个执行动作绝对值越大，惩罚越大，减少抖动 |

| 步数惩罚 | 步数越多惩罚越大，鼓励更快完成 |

  

环境层 `RobotRun2.run()` 也会在触碰成功时给出较强奖励：

  

```python

reward = 50.0 + 80.0 * closeness_ratio

```

  

但下载版训练循环又叠加了自定义 reward，使训练更关注“动作后落点距离”和“偏踩处理”。

  

## 经验存储与学习

  

每一步动作后，如果样本有效且 `train_tai=True`，会存储经验：

  

```python

tai_agent.store_transition(

    state=[obs_tensor, tai_state, tai_state],

    discrete_action=tai_discrete_action,

    continuous_action=tai_continuous_action,

    reward=reward,

    next_state=[next_obs_tensor, next_tai_state, next_tai_state],

    done=done,

    value=tai_value,

    discrete_log_prob=tai_dict['discrete_log_prob'],

    continuous_log_prob=tai_continuous_log_prob,

)

```

  

注意这里保存的是：

  

```text

图像分支：obs_tensor

状态分支：tai_state

图/结构分支：tai_state

```

  

回合结束后，如果允许训练，会调用：

  

```python

loss_discrete, loss_continuous = tai_agent.learn()

```

  

也就是说，踩踏策略同时学习离散门控和连续三关节动作。

  

## 成功后的流程边界

  

`PPO_tai_episoid()` 中会记录左脚实际执行增量：

  

```python

real_du = post_robot_state_for_replay[11] - robot_state[11]

real_dl = post_robot_state_for_replay[13] - robot_state[13]

real_da = post_robot_state_for_replay[15] - robot_state[15]

left_step_history.append((real_du, real_dl, real_da))

```

  

这里记录的是动作执行后的真实关节变化，而不是原始策略输出。这样做是为了右脚复现时更贴近 Webots 实际执行结果，避免限位、到位误差造成左右不一致。

  

如果 `keep_pose_on_success=True` 且踩踏成功，会保存：

  

```python

env._last_tai_left_step_history = list(left_step_history)

```

  

然后主流程中：

  

```python

left_step_history = getattr(env, '_last_tai_left_step_history', None)

_reset_right_leg_pose_like_env_reset(env)

right_step_success = _replay_right_step_from_left_history(env, left_step_history)

```

  

这说明完整流程边界是：

  

| 阶段 | 内容 | 是否属于 `PPO_tai_episoid()` 核心训练 |

| --- | --- | --- |

| 左脚踩踏训练 | PPO 控制左腿三个关节，让左脚踩到目标台阶 | 是 |

| 右脚复现 | 根据左脚真实增量镜像/复现右脚动作 | 否 |

| 双脚站稳 | 双腿伸直并等待稳定 | 否 |

| 台阶上抓取 | 进入第三阶段训练 | 否 |

  

所以下载版当前训练的核心不是完整双脚踩踏流程，而是“左脚踩到目标台阶”这个子任务。右脚复现和后续上台阶抓取，是踩踏成功后的流程衔接。

  

## 与当前仓库版的逻辑差异

  

这一节只比较逻辑，不比较文件组织和函数拆分。总体上，两版都把踩踏理解为“左脚三关节控制后踩到 `gps_goal1` 附近”，不是单纯抬腿；差异主要在状态输入、动作映射、奖励位置和成功后的流程衔接。

  

| 对比点 | 下载版逻辑 | 当前仓库版逻辑 |

| --- | --- | --- |

| 训练目标 | 左脚踩到目标台阶，成功后可保留姿态进入右脚复现和三阶段 | 左脚踩到目标台阶，但 `run_tai_stage()` 结束后直接重置环境 |

| 进入条件 | `PPO_tai_episoid()` 要求 `catch_success=True` | `run_tai_stage()` 同样要求 `catch_success=True` |

| 踩踏状态输入 | 构造专门的 `tai_state`，包含腿部关节、关节变化、脚部 GPS/目标差值、IMU、左脚触碰、上一动作、上一门控 | 直接用 `robot_state` 作为状态/图分支输入，未构造下载版那种踩踏特征状态 |

| 图像输入 | `obs_tensor` 与 `tai_state` 一起输入 | `obs_img/obs_tensor` 与 `robot_state` 一起输入；经验里保存的是 `obs_img` |

| 动作输出 | 三维离散门控 + 三维连续动作，对应 `LegUpperL / LegLowerL / AnkleL` | 同样是三维离散门控 + 三维连续动作 |

| 门控逻辑 | 离散为 `0` 时保持上一执行动作，为 `1` 时使用新动作 | 同样保留这个门控逻辑 |

| 动作到关节目标 | 下载版 `RobotRun2 (2).py` 基本是 `当前关节角度 + action` | 当前仓库 `RobotRun2.py` 先把 action 映射为 delta：`1.09*a+0.59`、`1.14*a-1.11`、`1.305*a-0.085`，再加到当前关节角度 |

| 到位等待 | `_wait_leg_target_reached()` 等待三个左腿关节接近目标位 | 同样保留 `_wait_leg_target_reached()`，并把它封装得更清楚 |

| 成功判定 | 环境层和训练循环层双重要求：左脚触碰 + 距离 `< 0.04` | 环境层要求左脚触碰 + 距离 `<= 0.04`，成功时返回 `goal=1` |

| 奖励计算位置 | 下载版在 `PPO_tai_episoid()` 中重新计算一套稠密奖励，同时环境层也给 reward | 当前仓库主要把奖励集中到 `RobotRun2._compute_reward()`，`run_tai_stage()` 直接使用 `env.step2()` 返回的 `reward_env` |

| 左脚历史记录 | 记录动作后真实关节变化量，用于右脚复现 | 记录门控后的动作值，但当前主流程没有继续执行右脚复现 |

| 成功后处理 | `keep_pose_on_success=True` 时保留姿态，保存 `_last_tai_left_step_history`，主流程继续右脚复现 | `done` 后重置左腿、机器人和环境；`catch_success` 置回 `False`，重新开始后续决策/抓取 |

  

### 1. 状态输入差异最大

  

下载版更像“专门为踩踏设计的状态输入”。`_build_tai_feature_state()` 会把脚到目标的距离、横向偏差、IMU、左脚触碰、上一动作、上一门控都喂给策略。这样策略能直接看到：

  

```text

左脚现在在哪里

距离目标还差多少

是否已经触碰

身体姿态是否稳定

上一拍刚执行了什么

```

  

当前仓库版 `run_tai_stage()` 仍主要使用：

  

```python

tai_dict = tai_agent.choose_action(

    obs=[obs_img, robot_state],

    x_graph=robot_state,

)

```

  

也就是说，仓库版踩踏策略能看到图像和关节状态，但没有显式构造下载版那种“脚到台阶目标误差 + 触碰 + IMU + 上一步动作”的 40 维踩踏特征。逻辑上，仓库版更依赖环境奖励把策略推向正确动作；下载版则把更多“踩踏相关事实”直接作为状态输入。

  

### 2. 动作语义相同，但映射不同

  

两版策略输出的动作语义一致：都是控制左腿三个关节。

  

```text

LegUpperL / LegLowerL / AnkleL

```

  

但动作变成关节目标的方式不同。

  

下载版环境层更接近：

  

```text

目标角度 = 当前关节角度 + 策略动作

```

  

当前仓库版在 `RobotRun2.py` 中先把动作映射成带比例和偏置的增量：

  

```python

self.leg_upper_delta = 1.09 * self.action_leg_upper + 0.59

self.leg_lower_delta = 1.14 * self.action_leg_lower - 1.11

self.ankle_delta = 1.305 * self.action_ankle - 0.085

```

  

再计算：

  

```python

target = 当前关节角度 + delta

```

  

所以逻辑差异是：下载版的连续动作更像“直接增量”，仓库版的连续动作先经过线性变换，动作的零点和尺度已经被重新定义。

  

### 3. 奖励逻辑的位置不同

  

下载版奖励有两层：

  

1. `RobotRun2 (2).py` 环境层先按距离、触碰、偏踩给一个 reward。

2. `PPO_episoid_2_1 (2).py` 的训练循环里又基于动作后 GPS 重新计算稠密 reward，包括距离惩罚、进度奖励、中心误差、成功奖励、偏踩惩罚、动作惩罚、步数惩罚。

  

当前仓库版把这些踩踏奖励逻辑更多集中到了 `RobotRun2._compute_reward()`：

  

```text

距离惩罚

距离进度奖励

中心误差/横向偏差惩罚

成功奖励

偏踩惩罚

无触碰惩罚

动作幅度惩罚

步数惩罚

碰撞惩罚

```

  

`run_tai_stage()` 本身不再另写一套稠密 reward，而是直接使用：

  

```python

reward = float(reward_env)

```

  

因此，下载版是“训练循环主导 reward，环境层也参与”；仓库版是“环境层主导 reward，训练循环只存储和学习”。

  

### 4. 成功判定逻辑大体一致

  

两版的核心成功语义一致：

  

```text

左脚触碰台阶 + 动作后 foot_gps1 到 gps_goal1 的距离足够小

```

  

下载版环境层使用：

  

```python

total_distance < self.success_dist_thresh

```

  

训练循环层又用：

  

```python

strict_success = (goal == 1 and dist < TAI_SUCCESS_DIST_THRESH)

```

  

当前仓库版环境层使用：

  

```python

if distance <= self.success_dist_thresh:

    goal = 1

```

  

阈值都是 `0.04`。差异只在边界符号和兜底层级：下载版在训练循环里又做了一次严格校验，仓库版主要以 `RobotRun2.run()` 返回的 `goal` 为准。

  

### 5. 成功后的流程差异很关键

  

下载版踩踏成功后，如果 `keep_pose_on_success=True`，会保留当前姿态，并保存：

  

```python

env._last_tai_left_step_history = list(left_step_history)

```

  

主流程随后用这段历史做右脚复现：

  

```python

right_step_success = _replay_right_step_from_left_history(env, left_step_history)

```

  

也就是说，下载版把左脚踩踏训练嵌进了更完整的“左脚上台阶 -> 右脚复现 -> 双脚稳定 -> 台阶上抓取”流程。

  

当前仓库版 `run_tai_stage()` 在回合结束后会：

  

```python

env.darwin._set_left_leg_initpose()

env.darwin.robot_reset()

env.reset()

catch_success = False

```

  

所以仓库版目前更像“单独训练/验证左脚踩踏阶段”，没有把踩踏成功后的姿态继续传给右脚复现和第三阶段。

  

### 6. 左脚历史记录的含义不同

  

下载版记录的是动作执行后的真实关节变化：

  

```text

post_robot_state - robot_state

```

  

这适合后续右脚复现，因为它记录的是 Webots 实际到达后的效果。

  

当前仓库版记录的是门控后的执行动作：

  

```python

left_step_history.append((action_leg_upper, action_leg_lower, action_leg_ankle))

```

  

而且当前仓库主流程没有继续使用它做右脚复现。因此仓库版的 `left_step_history` 更像保留接口/调试信息，下载版的 `left_step_history` 是后续流程真正依赖的数据。

  

### 7. 总结差异

  

如果只看逻辑，不看代码结构，可以这样概括：

  

```text

下载版：

  更完整的踩踏闭环。

  状态输入更丰富。

  reward 在训练循环里更强干预。

  成功后保留姿态并衔接右脚复现。

  

当前仓库版：

  保留了左脚踩踏的核心执行与成功判定。

  环境层 RobotRun2 更模块化，reward 更集中。

  状态输入比下载版简化。

  成功后仍重置环境，没有继续右脚复现流程。

```

  

因此，两版不是“一个训练抬腿、一个训练踩踏”的区别；两版概念上都在训练左脚踩踏。真正差异是：下载版更偏完整流程集成，仓库版更偏把左脚踩踏单阶段抽出来训练，并把环境执行/奖励逻辑整理到了 `RobotRun2.py`。

  

## 名词区分

  

| 名词 | 含义 |

| --- | --- |

| `target` / `final_target` | 三个左腿关节的目标角度 |

| `gps_goal1` | 目标台阶位置坐标 |

| `foot_gps1` | 左脚动作后的 GPS 位置 |

| `goal` | 环境返回的本步是否踩踏成功标志 |

| `done` | 当前踩踏回合是否结束 |

| `left_step_history` | 左脚真实执行增量历史，用于右脚复现 |

| `TAI_SUCCESS_DIST_THRESH = 0.04` | 训练循环层严格成功距离阈值 |

  

最容易混淆的是 `target` 和 `gps_goal1`：前者是关节角度目标，后者才是空间中的台阶目标位置。