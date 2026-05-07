import os

import torch


class RobotRun0:
    """Wrapper for the upper decision agent."""

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

    def choose_action(self, decision_state_input, pressure_detected=None):
        print(f"decision_pressure_state: {decision_state_input}")
        if pressure_detected is not None:
            print(f"pressure_detected={pressure_detected}")

        decision_dict = self.decision_agent.choose_action(obs=decision_state_input)
        decision = int(decision_dict["discrete_action"][0])
        print(f"upper decision = {decision} (0=catch, 1=tai)")

        decision_state = (decision_state_input, decision_state_input, decision_state_input)
        return decision_dict, decision, decision_state

    def judge_route(self, decision, catch_success):
        if decision == 0:
            if catch_success:
                return {
                    "route": "re_decide",
                    "reason": "already_caught_keep_pose",
                }
            return {
                "route": "catch",
                "reason": "need_catch",
            }

        if catch_success:
            return {
                "route": "tai",
                "reason": "caught_ready_for_tai",
            }

        return {
            "route": "re_decide",
            "reason": "not_caught_cannot_tai",
        }

    def _compute_reward(self, decision, route, pre_branch_catch_success, post_branch_catch_success):
        if decision == 0:
            if pre_branch_catch_success:
                print("Decision error: already caught but selected catch again. reward=-15.0")
                return -15.0
            if post_branch_catch_success:
                print("Decision correct: selected catch and catch succeeded. reward=+5.0")
                return 5.0
            print("Decision failed: selected catch but catch failed. reward=-8.0")
            return -8.0

        if pre_branch_catch_success:
            print("Decision correct: already caught and selected tai. reward=+10.0")
            return 10.0

        print("Decision error: not caught but selected tai. reward=-10.0")
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
            action=[float(decision)],
            reward=decision_reward,
            next_state=None,
            done=True,
            value=decision_dict["value"],
            log_prob=decision_dict["discrete_log_prob"],
        )

        self.training_manager.increment_decision()
        if self.training_manager.should_learn_decision():
            decision_loss = self.decision_agent.learn()
            print(f"[decision learn] {self.training_manager.get_status()} | decision_loss: {decision_loss:.6f}")
        else:
            print(f"[decision buffer] {self.training_manager.get_status()}")
            decision_loss = 0

        self.log_writer_decision.add_cycle(
            total_episode=total_episode,
            decision_action=decision,
            loss_discrete=decision_loss,
            loss_continuous=0.0,
            decision_reward=decision_reward,
            route=route,
            pre_catch_success=pre_branch_catch_success,
            post_catch_success=post_branch_catch_success,
        )
        self.log_writer_decision.save(self.log_file_latest_decision)

        if total_episode % self.save_interval == 0:
            dec_path = os.path.join(self.decision_checkpoint_dir, f"decision_hppo_{total_episode}.ckpt")
            dec_ckpt = {
                "policy": self.decision_agent.policy.state_dict(),
                "optimizer": self.decision_agent.optimizer.state_dict(),
                "episode": total_episode,
                "state_dim": getattr(self.decision_agent, "state_dim", None),
                "use_image_input": getattr(self.decision_agent, "use_image_input", None),
            }
            torch.save(dec_ckpt, dec_path)

        return decision_reward, decision_loss
