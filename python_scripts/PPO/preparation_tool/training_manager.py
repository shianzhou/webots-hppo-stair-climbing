class TrainingManager:
    """
    统一管理三个模型（抓取、抬腿、决策）的训练节奏，防止梯度聚集和模型冲突。
    核心思想：不同模型以不同频率进行学习，避免同时大量参数更新。
    """

    def __init__(self):
        self.catch_episodes = 0
        self.tai_episodes = 0
        self.decision_episodes = 0

        self.catch_learn_interval = 3
        self.tai_learn_interval = 2
        self.decision_learn_interval = 5

        print("【训练管理器初始化】")
        print(f"  抓取学习间隔: {self.catch_learn_interval}个episodes")
        print(f"  抬腿学习间隔: {self.tai_learn_interval}个episodes")
        print(f"  决策学习间隔: {self.decision_learn_interval}个episodes")

    def should_learn_catch(self) -> bool:
        """决定是否让抓取模型学习（降低频率防止梯度聚集）"""
        return (self.catch_episodes % self.catch_learn_interval == 0) and (self.catch_episodes > 0)

    def should_learn_tai(self) -> bool:
        """决定是否让抬腿模型学习（更新频率较低，批量大）"""
        return (self.tai_episodes % self.tai_learn_interval == 0) and (self.tai_episodes > 0)

    def should_learn_decision(self) -> bool:
        """决定是否让决策模型学习（最稀疏的更新）"""
        return (self.decision_episodes % self.decision_learn_interval == 0) and (self.decision_episodes > 0)

    def increment_catch(self):
        """抓取episode计数加1"""
        self.catch_episodes += 1

    def increment_tai(self):
        """抬腿episode计数加1"""
        self.tai_episodes += 1

    def increment_decision(self):
        """决策episode计数加1"""
        self.decision_episodes += 1

    def get_status(self) -> str:
        """获取当前训练状态"""
        return (
            f"[TrainingManager] Catch:{self.catch_episodes} | "
            f"Tai:{self.tai_episodes} | Decision:{self.decision_episodes}"
        )
