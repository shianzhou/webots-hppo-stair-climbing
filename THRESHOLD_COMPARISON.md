# 约束阈值详细对比：旧版本 vs 新版本

## 1. IMU 和关节限制阈值（配置值）

**完全相同**：
```
acc_low   = [480, 450, 580]
acc_high  = [560, 530, 700]
gyro_low  = [500, 500, 500]
gyro_high = [520, 520, 520]
```

---

## 2. 跟踪容差对比

| 检查项 | 旧版本 | 新版本 | 变化 |
|-------|-------|-------|------|
| **普通跟踪容差** | 0.005 | 0.005 (基础) | 相同 |
| **等待动作容差** | - | 0.01 | ✅ 新增（更宽松）|
| **约束检查容差** | 0.005 | 0.02 | ⚠️ 变宽松了4倍 |

**旧版本跟踪（第335行）**：
```python
if -0.005 < self.cha_zhi < 0.005:
    continue
else:
    # 终止，done=1
```

**新版本跟踪（第226、385行）**：
```python
def _wait_action_settle(self, max_steps=100, tol=0.01):  # 等待时用 0.01
    ...

def _check_tracking_constraint(self, tol=0.005):  # 定义时用 0.005
    ...

# 但调用时：
tracking_ok = self._check_tracking_constraint(tol=0.02)  # ⚠️ 实际用 0.02
```

---

## 3. 约束检查顺序对比（这是关键差异！）

### 旧版本执行流程：
```
1. 执行 IMU 检查 → 直接返回
2. 执行关节角度检查（2次）→ 直接返回
3. 执行上肢动作
4. 等待舵机到位（100次循环，容差 0.01）
5. 检查碰撞传感器
6. 如果没碰撞，检查抓爪触碰
7. 如果都没触碰，**最后**检查跟踪偏差（容差 0.005）
```

### 新版本执行流程：
```
1. 计算距离评分
2. 执行关节角度检查
3. 执行 IMU 检查 → 直接返回（约束1）
4. 执行上肢动作
5. 等待动作结束（容差 0.01）
6. 检查碰撞约束 → 直接返回（约束2）
7. 检查抓爪触碰
8. 如果都没触碰，**立即**检查跟踪偏差（容差 0.02） → 直接返回（约束5）
```

---

## 4. 问题诊断

### 新版本为什么显得"更敏感"？

不是因为数值阈值严苛，而是：

1. **检查顺序改变**：旧版本的跟踪检查是**最后**一步，新版本是**中间**一步
2. **容差反而增大了**：从 0.005 → 0.02（4倍宽松）
3. **但多了一个"等待动作"步骤**：这个步骤可能导致额外的延迟和不确定性

### 真正的严苛之处：
- **不是跟踪容差**，而是**碰撞约束立即返回**
- 旧版本中碰撞检查是在非if/else分支中，新版本是硬约束
- 新版本如果任一碰撞传感器 == 1.0 就立即 done=1

---

## 5. 关键阈值列表

### 关节限制（所有 20 个舵机）
```python
limit = [
    [-3.14, 3.14],   # 舵机 0
    [-3.14, 2.85],   # 舵机 1
    [-0.68, 2.30],   # 舵机 2
    ...
    [-0.36, 0.94]    # 舵机 19
]
```
**检查方式**：`low <= future_state[i] <= high`（使用future_state，还未执行）

### 碰撞传感器（6个）
```python
touch_peng = [
    arm_L1, arm_R1,      # 手臂传感器（2个）
    leg_L1, leg_L2,      # 左腿传感器（2个）
    leg_R1, leg_R2       # 右腿传感器（2个）
]
```
**检查方式**：`getValue() == 1.0`（严格等于1.0）

### IMU（加速度 + 陀螺仪）
```python
acc_low[i] < acc[i] < acc_high[i]       # 严格不等号
gyro_low[i] < gyro[i] < gyro_high[i]    # 严格不等号
```

---

## 6. 立即行动建议

### 方案 A：放宽跟踪容差（推荐）
当前新版本已经是 0.02，可以尝试进一步放宽：

```python
# 当前
tracking_ok = self._check_tracking_constraint(tol=0.02)

# 改为
tracking_ok = self._check_tracking_constraint(tol=0.03)
```

### 方案 B：改进碰撞检查敏感度
```python
def _check_collision_constraint(self):
    """碰撞约束：检查碰撞传感器触发情况。"""
    collision_count = sum(1 for sensor in self.touch_peng if sensor.getValue() == 1.0)
    return collision_count >= 2  # 至少2个才算碰撞（而不是任1个）
```

### 方案 C：检查 IMU 值是否真的超出范围
添加日志看看实际的 IMU 值：

```python
def _check_imu_limits(self):
    acc = self.accelerometer.getValues()
    gyro = self.gyro.getValues()
    print(f"  [IMU] acc={acc} | gyro={gyro}")
    print(f"  [IMU Range] acc should be in {Darwin_config.acc_low} ~ {Darwin_config.acc_high}")
    print(f"  [IMU Range] gyro should be in {Darwin_config.gyro_low} ~ {Darwin_config.gyro_high}")
    for i in range(3):
        if not (Darwin_config.acc_low[i] < acc[i] < Darwin_config.acc_high[i] and
                Darwin_config.gyro_low[i] < gyro[i] < Darwin_config.gyro_high[i]):
            return False
    return True
```

---

## 7. 总结

| 问题 | 旧版本 | 新版本 | 建议 |
|------|-------|-------|------|
| **跟踪容差** | 0.005 | 0.02 | ✅ 已改善，可进一步放宽到 0.03 |
| **IMU 阈值** | 480-560, 500-520 | 480-560, 500-520 | 无变化，检查实际值 |
| **关节限制** | 相同 | 相同 | 相同 |
| **碰撞敏感度** | 触发一个传感器 | 触发一个传感器 | 考虑改为需要多个触发 |
| **检查顺序** | 先动作后约束 | 先约束后动作 | ⚠️ 这导致提前终止 |

**最可能的问题**：不是阈值本身，而是**新版本改动了约束检查时机**（在动作前就检查），导致valid episode提前结束。

