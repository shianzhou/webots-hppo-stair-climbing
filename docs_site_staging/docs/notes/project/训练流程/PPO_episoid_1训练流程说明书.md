# PPO_episoid_1 训练流程说明书

对应代码：`E:\project_MultiAgent_h_change\python_scripts\PPO\PPO_episoid_1.py`

本文说明当前主训练脚本 `PPO_episoid_1.py` 的运行逻辑。它是一个三阶段分层 PPO 训练入口：

```text
上层决策 RobotRun0
  -> 决定本轮走抓取 catch、踩踏 tai，还是重新决策
抓取阶段 run_catch_stage
  -> 控制上肢抓取目标
踩踏阶段 run_tai_stage
  -> 控制左腿三关节踩到目标台阶
```

当前结构里只有一个 Webots `Environment` 实例，三个模型围绕同一个环境交替执行。

## 1. 文件依赖关系

`PPO_episoid_1.py` 主要依赖：

| 模块 | 作用 |
| --- | --- |
| `checkpoint_utils.py` | 创建目录、加载 checkpoint、查找最新日志文件 |
| `RobotRun0.py` | 上层决策包装器，负责路线选择和决策层训练 |
| `training_manager.py` | 控制各阶段什么时候触发 `learn()` |
| `hppo.py` | 决策层 PPO，当前用于 6 维纯状态输入 |
| `hppo_01.py` | 抓取/踩踏 PPO，支持离散+连续动作 |
| `log_code` | 三阶段日志写入 |
| `state_inputs.py` | 三阶段状态输入构造 |
| `Project_config.path_list` | 模型路径、日志路径等配置 |
| `Webots_interfaces.Environment` | Webots 环境接口 |

## 2. 三个模型的初始化

入口函数先调用：

```python
init_training_and_logging(path_list)
```

这个函数统一创建训练管理器、三个 PPO agent、日志写入器、checkpoint 目录和日志路径。

### 2.1 抓取模型

```python
hppo_agent = hppo(
    num_servos=2,
    node_num=19,
    state_dim=CATCH_STATE_DIM,
    use_image_input=True,
)
```

含义：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `num_servos` | 2 | 抓取控制肩部和手臂两个动作通道 |
| `state_dim` | 20 | 抓取使用 20 维关节状态 |
| `use_image_input` | True | 抓取仍使用图像分支 |

抓取模型输出：

```text
2 维 discrete_action + 2 维 continuous_action
```

### 2.2 踩踏模型

```python
tai_agent = hppo(
    num_servos=3,
    node_num=19,
    state_dim=TAI_STATE_DIM,
    use_image_input=False,
)
```

含义：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `num_servos` | 3 | 控制 `LegUpperL / LegLowerL / AnkleL` |
| `state_dim` | 40 | 踩踏使用 40 维专用状态 |
| `use_image_input` | False | 踩踏不再使用图像 |

踩踏模型输出：

```text
3 维 discrete_action + 3 维 continuous_action
```

### 2.3 决策模型

```python
decision_hppo_agent = d_hppo(
    num_servos=1,
    node_num=19,
    state_dim=DECISION_STATE_DIM,
    use_image_input=False,
)
```

含义：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `num_servos` | 1 | 只输出一个上层决策 |
| `state_dim` | 6 | 6 维手部压力状态 |
| `use_image_input` | False | 决策层不用图像 |

决策模型输出：

```text
1 维 discrete_action，0=catch，1=tai
```

## 3. 状态输入来源

三阶段状态输入都来自 `state_inputs.py`。

| 阶段 | 构造函数 | 输入维度 | 是否使用图像 |
| --- | --- | --- | --- |
| 决策 | `build_decision_state(env)` | 6 | 否 |
| 抓取 | `build_catch_state(env, step)` | 图像 + 20 | 是 |
| 踩踏 | `build_tai_state(...)` | 40 | 否 |

### 3.1 决策状态

主循环中：

```python
d_pressure_state, d_pressure_values, pressure_detected = build_decision_state(env)
```

状态内容：

```text
grasp_L1, grasp_L1_1, grasp_L1_2, grasp_R1, grasp_R1_1, grasp_R1_2
```

用途：让上层决策知道当前手部压力状态。

### 3.2 抓取状态

抓取阶段中：

```python
obs_img, obs_tensor, robot_state, obs, x_graph = build_catch_state(env, steps)
```

其中：

- `obs_tensor`：图像分支输入。
- `robot_state`：20 维关节状态。
- `x_graph=robot_state`：图/结构分支输入。

抓取仍是视觉 + 关节状态。

### 3.3 踩踏状态

踩踏阶段中：

```python
tai_state = build_tai_state(
    env=env,
    robot_state=robot_state,
    prev_robot_state=prev_robot_state,
    prev_exec_actions=prev_exec_actions,
    prev_discrete_actions=prev_discrete_actions,
    feature_dim=TAI_STATE_DIM,
)
```

踩踏 40 维状态包含：

- 腿部关键关节角。
- 关节变化量。
- 左脚 GPS 与 `gps_goal1` 的差值。
- IMU。
- 左脚触碰。
- 上一步执行动作。
- 上一步离散门控。

踩踏阶段不再调用 `env.get_img()`，不使用图像。

## 4. 主入口 `PPO_episoid_1()`

主入口：

```python
def PPO_episoid_1(model_path=None, max_steps_per_episode=20):
```

运行顺序：

```text
init_training_and_logging()
  -> 取出 training_manager / 三个 agent / 三个 log writer / checkpoint 路径
  -> load_catch_model()
  -> load_tai_model()
  -> load_decision_model()
  -> 创建 Environment
  -> env.reset()
  -> 创建 RobotRun0
  -> while total_episode < MAX_TOTAL_EPISODE:
       build_decision_state()
       RobotRun0.choose_action()
       RobotRun0.judge_route()
       根据 route 调用抓取或踩踏
       RobotRun0.finalize()
       total_episode += 1
```

当前最大总周期：

```python
MAX_TOTAL_EPISODE = 10000
```

## 5. 模型加载逻辑

主入口加载三个模型：

```python
load_catch_model(model_path, hppo_agent, path_list['model_path_catch_PPO_h'])
load_tai_model(tai_agent, path_list['model_path_tai_PPO_h'], default_episode=tai_episoid)
load_decision_model(decision_hppo_agent, path_list.get('model_path_decision_PPO_h'))
```

### 5.1 抓取 checkpoint

抓取模型从 catch 路径加载。抓取仍是图像 + 20 维状态。

### 5.2 踩踏 checkpoint

踩踏模型现在是：

```text
state_dim=40, use_image_input=False
```

因此旧的视觉踩踏模型或旧 20 维 checkpoint 可能不兼容。加载逻辑会检查维度和 `use_image_input`，不兼容时跳过，从当前网络重新训练。

### 5.3 决策 checkpoint

决策模型是：

```text
state_dim=6, use_image_input=False
```

不兼容时同样跳过加载。

## 6. 上层决策流程

每个 `total_episode` 都先执行决策：

```python
d_pressure_state, d_pressure_values, pressure_detected = build_decision_state(env)
decision_dict, decision, decision_state = robot_run0.choose_action(
    d_pressure_state,
    pressure_detected=pressure_detected,
)
```

然后根据当前抓取状态判断路线：

```python
pre_branch_catch_success = catch_success
route_info = robot_run0.judge_route(decision=decision, catch_success=pre_branch_catch_success)
route = route_info['route']
```

路线有三种：

| route | 含义 |
| --- | --- |
| `catch` | 执行抓取训练阶段 |
| `tai` | 执行踩踏训练阶段 |
| `re_decide` | 不执行子阶段，直接重新决策 |

路由判断核心：

```text
decision=0 且未抓取 -> catch
decision=0 且已抓取 -> re_decide
decision=1 且已抓取 -> tai
decision=1 且未抓取 -> re_decide
```

## 7. 抓取阶段 `run_catch_stage()`

函数签名：

```python
def run_catch_stage(
    env,
    hppo_agent,
    log_writer_catch,
    log_file_latest_catch,
    catch_checkpoint_dir,
    catch_save_interval,
    total_episode,
    episode_num,
    max_steps_per_episode,
):
```

### 7.1 进入前准备

抓取阶段开始时会：

```python
env.reset()
env.wait(500)
```

也就是说，每次抓取阶段都从环境重置后的状态开始。

### 7.2 每一步执行

循环中：

```python
obs_img, obs_tensor, robot_state, obs, x_graph = build_catch_state(env, steps)
action_dict = hppo_agent.choose_action(obs=obs, x_graph=x_graph)
```

模型输出：

```python
d_action = action_dict['discrete_action']
c_action = action_dict['continuous_action']
```

执行环境动作：

```python
next_state, reward, done, catch_success = env.step(d_action, c_action, steps)
```

这里的 `env.step()` 内部对应抓取阶段的 `RobotRun1` 执行。

### 7.3 经验缓存与学习

每一步后都会保存经验：

```python
hppo_agent.store_transition(...)
```

然后立即调用：

```python
loss_d, loss_c = hppo_agent.learn()
```

注意：当前抓取阶段是每步都尝试 learn。`hppo_01.py` 内部如果 buffer 不足，会直接返回 `0, 0`。

### 7.4 结束与保存

结束条件：

```python
if done == 1 or steps >= max_steps_per_episode:
    break
```

结束后写抓取日志：

```python
log_writer_catch.add_episode(...)
log_writer_catch.save(log_file_latest_catch)
```

如果满足保存间隔：

```python
hppo_agent.save_checkpoint(catch_path, episode=total_episode)
```

返回：

```python
return catch_success
```

这个返回值会成为主循环中的跨 episode 状态。

## 8. 踩踏阶段 `run_tai_stage()`

函数签名：

```python
def run_tai_stage(
    env,
    tai_agent,
    training_manager,
    log_writer_tai,
    log_file_latest_tai,
    tai_checkpoint_dir,
    total_episode,
    tai_episoid,
    catch_success,
    max_steps=20,
    save_interval=400,
):
```

### 8.1 前置条件

踩踏必须要求：

```python
if not catch_success:
    return tai_episoid, catch_success
```

也就是说，只有抓取成功后才允许进入踩踏。

### 8.2 准备动作

进入踩踏后会先执行：

```python
env.darwin.tai_leg_L1()
env.darwin.tai_leg_L2()
```

这是踩踏前的预备姿态动作。

### 8.3 每一步状态构造

踩踏不使用图像。每步读取当前关节状态：

```python
robot_state = validate_and_clean_data(env.get_robot_state())
```

构造 40 维踩踏状态：

```python
tai_state = build_tai_state(...)
```

### 8.4 模型动作

调用：

```python
tai_dict = tai_agent.choose_action(
    obs=tai_state,
    x_graph=tai_state,
)
```

输出包括：

```python
tai_discrete_action
tai_continuous_action
tai_continuous_log_prob
tai_value
```

三维动作对应：

| 维度 | 关节 |
| --- | --- |
| 0 | `LegUpperL` |
| 1 | `LegLowerL` |
| 2 | `AnkleL` |

离散门控逻辑：

```python
action = raw_action if discrete_mask == 1 else prev_exec_action
```

也就是离散为 0 时保持上一执行动作。

### 8.5 执行动作

先读取 GPS：

```python
gps_values = validate_and_clean_data(env.print_gps())
```

再调用：

```python
next_state, reward_env, done, good, goal, count = env.step2(...)
```

这里的 `env.step2()` 内部对应 `RobotRun2`，负责把三关节动作映射为电机目标并判断踩踏结果。

### 8.6 经验缓存

动作后构造下一时刻踩踏状态：

```python
next_tai_state = build_tai_state(...)
```

若样本有效，则保存经验：

```python
tai_agent.store_transition(
    state=[None, tai_state, tai_state],
    next_state=[None, next_tai_state, next_tai_state],
    ...
)
```

这里的 `None` 表示踩踏不使用图像分支。

### 8.7 学习与保存

若 `done == 1`：

1. 通过 `training_manager.increment_tai()` 记录踩踏训练次数。
2. 如果 `training_manager.should_learn_tai()` 为真，调用：

```python
loss_d, loss_c = tai_agent.learn()
```

3. 写入踩踏日志。
4. 按 `save_interval` 保存 checkpoint。

### 8.8 结束后处理

踩踏结束后会重置：

```python
env.darwin._set_left_leg_initpose()
env.darwin.robot_reset()
env.reset()
env.wait(1000)
```

然后：

```python
tai_episoid += 1
catch_success = False
return tai_episoid, catch_success
```

也就是说，当前仓库版踩踏完成后不会保留姿态继续右脚复现，而是重置环境并重新进入决策/抓取流程。

## 9. 决策层收尾 `RobotRun0.finalize()`

无论本轮路线是 `catch`、`tai` 还是 `re_decide`，主循环最后都会调用：

```python
robot_run0.finalize(...)
```

它会：

1. 根据本轮 `decision`、`route`、执行前后的 `catch_success` 计算决策奖励。
2. 保存决策经验。
3. 根据 `training_manager` 判断是否学习。
4. 写入决策日志。
5. 按间隔保存决策 checkpoint。

这意味着：上层决策模型并不是只在执行了抓取/踩踏后训练，`re_decide` 情况也会进入决策层奖励和训练逻辑。

## 10. 三阶段训练数据流

### 10.1 决策层

```text
build_decision_state(env)
  -> decision_agent.choose_action(obs=6维压力)
  -> RobotRun0.judge_route()
  -> RobotRun0.finalize()
  -> decision_agent.store_transition()
  -> decision_agent.learn()
```

### 10.2 抓取层

```text
build_catch_state(env, step)
  -> hppo_agent.choose_action(图像 + 20维关节)
  -> env.step(d_action, c_action, step)
  -> hppo_agent.store_transition()
  -> hppo_agent.learn()
```

### 10.3 踩踏层

```text
build_tai_state(...)
  -> tai_agent.choose_action(40维纯状态)
  -> env.step2(三关节动作)
  -> build_tai_state(...) 得到 next_tai_state
  -> tai_agent.store_transition()
  -> tai_agent.learn()
```

## 11. 日志与 checkpoint

### 11.1 日志文件

初始化时创建三个最新日志路径：

```python
catch_log_*.json
tai_log_*.json
decision_log_*.json
```

对应路径来自 `path_list`：

| 日志 | 配置 key |
| --- | --- |
| 抓取日志 | `catch_log_path_PPO` |
| 踩踏日志 | `tai_log_path_PPO` |
| 决策日志 | `decision_log_path_PPO` |

### 11.2 checkpoint 目录

| 模型 | 配置 key |
| --- | --- |
| 抓取 | `model_path_catch_PPO_h` |
| 踩踏 | `model_path_tai_PPO_h` |
| 决策 | `model_path_decision_PPO_h` |

### 11.3 保存时机

抓取：

```python
if done == 1 and total_episode % catch_save_interval == 0:
    hppo_agent.save_checkpoint(...)
```

踩踏：

```python
if done == 1 and total_episode % save_interval == 0:
    tai_agent.save_checkpoint(...)
```

决策：在 `RobotRun0.finalize()` 中按 `save_interval` 保存。

## 12. 当前训练流程的关键状态变量

| 变量 | 作用 |
| --- | --- |
| `total_episode` | 上层总循环计数 |
| `episode_num` | 抓取阶段 episode 计数变量，目前主流程未明显递增 |
| `tai_episoid` | 踩踏阶段 episode 计数 |
| `catch_success` | 跨阶段状态，表示上一轮是否抓取成功 |
| `pre_branch_catch_success` | 本轮分支执行前的抓取状态 |
| `post_branch_catch_success` | 本轮分支执行后的抓取状态 |
| `route` | 本轮真实执行路线 |

最重要的是 `catch_success`：

```text
False -> 决策只能合理进入 catch
True  -> 决策可以进入 tai
```

## 13. 当前流程的特点与注意点

1. 训练是上层决策驱动的，不是固定先抓取再踩踏。
2. 决策模型每个 `total_episode` 都会执行一次。
3. 抓取阶段每次进入前会 `env.reset()`。
4. 踩踏阶段必须依赖 `catch_success=True`。
5. 踩踏阶段不使用图像，只用 40 维 `tai_state`。
6. 踩踏结束后会重置环境，并把 `catch_success` 置回 `False`。
7. 当前仓库版还没有在踩踏成功后继续右脚复现或三阶段上台阶抓取。
8. 抓取和踩踏都使用 `hppo_01.py`，但输入形式不同：抓取有图像，踩踏无图像。
9. 决策使用 `hppo.py`，是纯 6 维状态输入。
10. 如果 checkpoint 输入维度或 `use_image_input` 不匹配，会跳过加载并从当前网络重新训练。

## 14. 简化版流程图

```text
启动 PPO_episoid_1
  |
  v
初始化三模型、日志、checkpoint 路径
  |
  v
加载已有 checkpoint
  |
  v
创建 Environment 并 reset
  |
  v
while total_episode < 10000:
  |
  v
  build_decision_state(env)
  |
  v
  RobotRun0.choose_action -> decision
  |
  v
  RobotRun0.judge_route(decision, catch_success)
  |
  +-- route == catch --> run_catch_stage()
  |
  +-- route == tai ----> run_tai_stage()
  |
  +-- route == re_decide -> 不执行子阶段
  |
  v
  RobotRun0.finalize()
  |
  v
  total_episode += 1
```

## 15. 如果后续要写测试/推理入口

测试入口可以复用本文件的结构，但应关闭：

```python
store_transition()
learn()
save_checkpoint()
```

也就是保留：

```text
状态构造 -> choose_action -> env.step/env.step2 -> 成功判定
```

而不进行训练更新。
