import os

import torch


class RobotRun0:
    """决策层封装：负责选动作、计算奖励、记录经验、学习和保存模型。"""

    def __init__(
        self,
        decision_agent,
        training_manager,
        log_writer_decision,
        log_file_latest_decision,
        decision_checkpoint_dir,
        save_interval=50,
    ):
        self.decision_agent = decision_agent
        self.training_manager = training_manager
        self.log_writer_decision = log_writer_decision
        self.log_file_latest_decision = log_file_latest_decision
        self.decision_checkpoint_dir = decision_checkpoint_dir
        self.save_interval = save_interval

    def choose_action(self, d_obs_tensor, d_robot_state):
        d_obs = (d_obs_tensor, d_robot_state)

        print(f"📊 d_obs_tensor shape: {d_obs_tensor.shape}, range: [{d_obs_tensor.min():.3f}, {d_obs_tensor.max():.3f}]")
        print(f"📊 d_robot_state shape: {len(d_robot_state) if isinstance(d_robot_state, list) else d_robot_state.shape}")

        decision_dict = self.decision_agent.choose_action(obs=d_obs, x_graph=d_robot_state)
        decision = int(decision_dict['discrete_action'][0])
        print(f"上层决策 decision = {decision} (0=抓取, 1=爬梯)")

        decision_state = (d_obs_tensor, d_robot_state, d_robot_state)
        return decision_dict, decision, decision_state

    def judge_route(self, decision, catch_success):
        """根据当前抓取状态判断决策动作是否可执行，以及应进入哪个阶段。"""
        if decision == 0:
            if catch_success:
                return {
                    'route': 're_decide',
                    'reason': 'already_caught_keep_pose',
                }
            return {
                'route': 'catch',
                'reason': 'need_catch',
            }

        if catch_success:
            return {
                'route': 'tai',
                'reason': 'caught_ready_for_tai',
            }

        return {
            'route': 're_decide',
            'reason': 'not_caught_cannot_tai',
        }

    def _compute_reward(self, decision, route, pre_branch_catch_success, post_branch_catch_success):
        if decision == 0:
            if pre_branch_catch_success:
                print("❌ 决策错误：已抓取成功仍选择抓取，保持动作并重新决策，惩罚-15.0")
                return -15.0
            if post_branch_catch_success:
                print("✅ 决策正确：未抓取状态选择抓取且抓取成功，奖励+5.0")
                return 5.0
            print("❌ 决策失败：选择抓取但抓取失败，重新决策，惩罚-8.0")
            return -8.0

        if pre_branch_catch_success:
            print("✅ 决策正确：已抓取状态选择抬腿，执行抬腿，奖励+10.0")
            return 10.0

        print("❌ 决策错误：未抓取状态选择抬腿，重新决策，惩罚-10.0")
        return -10.0

    def finalize(
        self,
        total_episode,
        decision,
        decision_dict,
        decision_state,
        route,
        pre_branch_catch_success,
        post_branch_catch_success,
    ):
        decision_reward = self._compute_reward(
            decision=decision,
            route=route,
            pre_branch_catch_success=pre_branch_catch_success,
            post_branch_catch_success=post_branch_catch_success,
        )

        self.decision_agent.store_transition(
            state=decision_state,
            action=decision,
            reward=decision_reward,
            next_state=None,
            done=True,
            value=decision_dict['value'],
            log_prob=decision_dict['discrete_log_prob'],
        )

        self.training_manager.increment_decision()
        if self.training_manager.should_learn_decision():
            decision_loss = self.decision_agent.learn()
            print(f'【决策模型学习】{self.training_manager.get_status()} | decision_loss: {decision_loss:.6f}')
        else:
            print(f'【决策模型累积经验】{self.training_manager.get_status()}')
            decision_loss = 0

        # 使用新的log_code系统的add_cycle()方法记录决策日志
        self.log_writer_decision.add_cycle(
            total_episode=total_episode,
            decision_action=decision,
            loss_discrete=decision_loss,
            loss_continuous=0.0,  # 决策层暂无连续损失
            decision_reward=decision_reward,
            route=route,
            pre_catch_success=pre_branch_catch_success,
            post_catch_success=post_branch_catch_success,
        )
        self.log_writer_decision.save(self.log_file_latest_decision)

        if total_episode % self.save_interval == 0:
            dec_path = os.path.join(self.decision_checkpoint_dir, f"decision_hppo_{total_episode}.ckpt")
            dec_ckpt = {
                'policy': self.decision_agent.policy.state_dict(),
                'optimizer': self.decision_agent.optimizer.state_dict(),
                'episode': total_episode,
            }
            torch.save(dec_ckpt, dec_path)

        return decision_reward, decision_loss
