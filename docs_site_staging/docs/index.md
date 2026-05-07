# webots-hppo-stair-climbing

这是一个面向 Webots 人形机器人、HPPO、三阶段强化学习训练流程的科研文档站点。

目标是把 Obsidian 中已经整理好的笔记，发布成类似 Spinning Up 风格的可检索文档：左侧章节清楚，正文适合公式、代码、流程图、实验图和训练日志长期沉淀。

## 文档入口

- [本地预览](getting-started/local-preview.md)
- [发布工作流](getting-started/publishing-workflow.md)
- [项目总览](notes/project/总体文件概览.md)
- [状态输入](notes/project/关于网络/状态输入.md)
- [训练结果](experiments/training-results.md)

## 示例

数学公式：

$$
J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T}\gamma^t r_t\right]
$$

Mermaid 流程图：

```mermaid
flowchart LR
    A[Webots Env] --> B[State Builder]
    B --> C[HPPO Policy]
    C --> D[RobotRun Layer]
    D --> A
```

代码块：

```python
state, pressure_values, pressure_detected = build_decision_state(env)
```
