# RobotRun1.py 版本对比：传感器灵敏度分析

## 核心差异总结

| 检查项 | 旧版本 (project_MultiAgent_h) | 新版本 (project_MultiAgent_h_change) | 影响 |
|-------|------|------|------|
| **传感器判定** | `np.array_equal([1.0, 1.0])` 严格匹配 | `touch_count >= 0.5` 宽松阈值 | ⚠️ 新版本**更敏感** |
| **碰撞检测** | `getValue() == 1.0` 严格等于 | `getValue() == 1.0` 严格等于 | 相同 |
| **跟踪容差** | `abs(diff) < 0.005` | `abs(diff) > tol (0.005)` | 相同逻辑 |
| **触碰判定阈值** | [1.0, 1.0] / [0.0, 0.0] | >= 0.5 or == 0 | ⚠️ 新版本**更宽松** |

---

## 传感器判定的关键差异

### 旧版本 (E:\project_MultiAgent_h)
```python
# 第 280-281 行
sucess = np.array_equal(self.touch_value, Darwin_config.touch_T)  # [1.0, 1.0]
faild = np.array_equal(self.touch_value, Darwin_config.touch_F)   # [0.0, 0.0]
```
**特点**：
- 只有当两个值**都正好是 1.0** 时才算成功
- 只有当两个值**都正好是 0.0** 时才算失败
- 其他值（如 [0.5, 0.8] 或 [0.2, 0.1]）**既不是成功也不是失败**，会被忽略

---

### 新版本 (E:\project_MultiAgent_h_change)
```python
# 第 210-213 行（当前代码）
touch_count = sum(1 for v in self.touch_value if v >= 0.5)
success = int(touch_count >= 2)  # 两个都>= 0.5
failed = int(touch_count == 0)   # 都< 0.5
```
**特点**：
- 只要两个值**都 >= 0.5** 就算成功（阈值更低）
- 只要**都 < 0.5** 就算失败
- 会捕获之前被忽略的"中间值"

---

## 问题分析

### 为什么"传感器太敏感"？

你看的日志：
```
catch_state=attempt_now | catch_success=False
```

这说明传感器读到的值**既不是 >= 0.5 也不是 < 0.5**（不可能），或者：
- 读到的是 [0.5, 0.3] → success=0（1个>= 0.5，不够2个）
- 读到的是 [0.7, 0.4] → success=0（1个>= 0.5，不够2个）

**新版本会把这些情况视为"部分接触"，而不是"失败"**，导致 `catch_state=partial_touch`，`catch_success` 永远为 False。

---

## 建议修复方案

### 方案 A: 恢复严格判定（回到旧版本逻辑）
```python
def _close_grasp_and_read_pair(self, wait_ms=2000):
    # ... 闭合代码 ...
    
    # 改回严格 array_equal 判定
    success = int(np.array_equal(self.touch_value, Darwin_config.touch_T))
    failed = int(np.array_equal(self.touch_value, Darwin_config.touch_F))
    
    print(f"  [传感器读取] touch_value={self.touch_value} | success={success} | failed={failed}")
    return success, failed
```
**优点**：与旧版本一致，已验证可行  
**缺点**：可能错过真正的成功（如果传感器值不完全是 1.0）

---

### 方案 B: 更智能的阈值判定（推荐）
```python
def _close_grasp_and_read_pair(self, wait_ms=2000):
    # ... 闭合代码 ...
    
    # 改为：任一值 >= 0.7（较严格的阈值）都算接触
    touch_count = sum(1 for v in self.touch_value if v >= 0.7)
    success = int(touch_count >= 1)  # 至少一个接触就算成功
    failed = int(all(v < 0.2 for v in self.touch_value))  # 都非常低才算失败
    
    print(f"  [传感器读取] touch_value={self.touch_value} | touch_count={touch_count} | success={success} | failed={failed}")
    return success, failed
```
**优点**：避免误触，也不会太严格  
**缺点**：需要根据实际调整阈值

---

### 方案 C: 检查旧版本的 touch_T 和 touch_F 是否配置正确
```python
# Project_config.py 中检查
print(f"Darwin_config.touch_T = {Darwin_config.touch_T}")
print(f"Darwin_config.touch_F = {Darwin_config.touch_F}")
```
可能旧版本中这两个值配置不同？

---

## 对比检查清单

- [ ] 旧版本的 `Darwin_config.touch_T` 是多少？
- [ ] 旧版本的 `Darwin_config.touch_F` 是多少？
- [ ] 旧版本运行时传感器实际读到的值通常是什么？
- [ ] 新版本中 Project_config.py 是否改过这些值？

---

## 立即测试步骤

1. **恢复严格判定**（方案 A）试试能否置真：

```python
# 第 210-215 行改为
success = int(np.array_equal(self.touch_value, Darwin_config.touch_T))
failed = int(np.array_equal(self.touch_value, Darwin_config.touch_F))
```

2. **观察日志**是否出现 `catch_success=True`

3. **如果仍不行**，检查传感器值日志，再根据实际值调整阈值

