# 测试
import torch

from python_scripts.PPO.PPO_episoid_2_1 import PPO_tai_episoid
from python_scripts.Webots_interfaces import Environment
from python_scripts.PPO.hppo_01 import HPPO as hppo
from python_scripts.PPO.hppo import HPPO as d_hppo
# from Data_fusion import data_fusion
from typing import Optional
from python_scripts.Project_config import path_list, gps_goal, gps_goal1, device
from python_scripts.PPO_Log_write import Log_write


# ===== 路径与文件工具函数（统一管理） =====
import os
import glob
import re


# ===== 方案A: 训练管理器（同步三个模型的更新频率） =====
class TrainingManager:
    """
    统一管理三个模型（抓取、抬腿、决策）的训练节奏，防止梯度聚集和模型冲突。
    核心思想：不同模型以不同频率进行学习，避免同时大量参数更新。
    """
    def __init__(self):
        self.catch_episodes = 0      # 抓取episodes计数
        self.tai_episodes = 0        # 抬腿episodes计数
        self.decision_episodes = 0   # 决策episodes计数
        
        # 【关键参数】控制各模型的学习频率
        self.catch_learn_interval = 3    # 每3个抓取episodes学习一次（抓取最频繁）
        self.tai_learn_interval = 2      # 每2个抬腿episodes学习一次（中等频率）
        self.decision_learn_interval = 5 # 每5个决策episodes学习一次（最稀疏）
        
        print("【训练管理器初始化】")
        print(f"  抓取学习间隔: {self.catch_learn_interval}个episodes")
        print(f"  抬腿学习间隔: {self.tai_learn_interval}个episodes")
        print(f"  决策学习间隔: {self.decision_learn_interval}个episodes")
    
    def should_learn_catch(self) -> bool:
        """决定是否让抓取模型学习（降低频率防止梯度聚集）"""
        result = (self.catch_episodes % self.catch_learn_interval == 0) and (self.catch_episodes > 0)
        return result
    
    def should_learn_tai(self) -> bool:
        """决定是否让抬腿模型学习（更新频率较低，批量大）"""
        result = (self.tai_episodes % self.tai_learn_interval == 0) and (self.tai_episodes > 0)
        return result
    
    def should_learn_decision(self) -> bool:
        """决定是否让决策模型学习（最稀疏的更新）"""
        result = (self.decision_episodes % self.decision_learn_interval == 0) and (self.decision_episodes > 0)
        return result
    
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
        return (f"[TrainingManager] Catch:{self.catch_episodes} | "
                f"Tai:{self.tai_episodes} | Decision:{self.decision_episodes}")


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

def _next_log_file(dir_path: str, prefix: str) -> str:
    pattern = os.path.join(dir_path, f"{prefix}_*.json")
    existing = glob.glob(pattern)
    max_n = 0
    for p in existing:
        m = re.search(rf"{re.escape(prefix)}_(\d+)\.json$", os.path.basename(p))
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except Exception:
                continue
    return os.path.join(dir_path, f"{prefix}_{max_n + 1}.json")

def _latest_catch_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "catch_hppo_*.ckpt"))
    if not files:
        return None, 0
    def _num(f: str) -> int:
        m = re.search(r"catch_hppo_(\d+)\.ckpt$", os.path.basename(f))
        return int(m.group(1)) if m else -1
    selected = max(files, key=_num)
    return selected, _num(selected)

def _latest_tai_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "tai_agent_*_*.ckpt"))
    if not files:
        return None, 0, 0
    def _nums(f: str):
        b = os.path.basename(f).replace('.ckpt','')
        parts = b.split('_')
        try:
            return int(parts[-2]), int(parts[-1])
        except Exception:
            return (0, 0)
    selected = max(files, key=_nums)
    total, ep = _nums(selected)
    return selected, total, ep

def _latest_decision_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "decision_hppo_*.ckpt"))
    if not files:
        return None, 0
    def _num(f: str) -> int:
        m = re.search(r"decision_hppo_(\d+)\.ckpt$", os.path.basename(f))
        return int(m.group(1)) if m else -1
    selected = max(files, key=_num)
    return selected, _num(selected)

# ===== 模型加载工具函数（提炼提高可读性） =====
def load_catch_model(model_path: str, hppo_agent, catch_dir: str) -> int:
    """加载抓取模型，优先使用指定路径；否则从目录中选择最新。返回 episode_start。"""
    episode_start = 0
    if model_path:
        try:
            ckpt = torch.load(model_path)
            if isinstance(ckpt, dict) and 'policy' in ckpt:
                hppo_agent.policy.load_state_dict(ckpt['policy'])
                if 'optimizer_hppo' in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt['optimizer_hppo'])
                print(f"从指定模型加载: {model_path}，模型加载成功！")
                try:
                    episode_start = int(os.path.basename(model_path).split('_')[-1].split('.')[0])
                    print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
                except Exception:
                    pass
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print(f"从指定模型加载: {model_path}，模型加载成功！(旧格式)")
        except Exception as e:
            print(f"指定模型加载失败: {e}")
            episode_start = 0
        return episode_start

    # 未指定路径，查找目录最新
    selected_model, episode_start = _latest_catch_ckpt(catch_dir)
    if selected_model:
        try:
            ckpt = torch.load(selected_model)
            if isinstance(ckpt, dict) and 'policy' in ckpt:
                hppo_agent.policy.load_state_dict(ckpt['policy'])
                if 'optimizer_hppo' in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt['optimizer_hppo'])
                print("抓取模型加载成功！")
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print("抓取模型加载成功！(旧格式)")
        except Exception as e:
            print(f"抓取模型加载失败: {e}")
            episode_start = 0
    else:
        print("未找到已保存的抓取模型，从头开始训练")
        episode_start = 0
    return episode_start

def load_tai_model(tai_agent, tai_dir: str, default_episode: int = 1) -> int:
    """加载抬腿模型，仅从新目录选择最新。返回抬腿起始回合。"""
    selected_tai, _, ep = _latest_tai_ckpt(tai_dir)
    if selected_tai:
        print(f"找到最新抬腿模型: {selected_tai}，抬腿周期: {ep}")
        try:
            ckpt = torch.load(selected_tai)
            if isinstance(ckpt, dict) and 'policy_tai' in ckpt:
                tai_agent.policy.load_state_dict(ckpt['policy_tai'])
                if 'optimizer_tai' in ckpt and tai_agent.optimizer:
                    tai_agent.optimizer.load_state_dict(ckpt['optimizer_tai'])
            print("抬腿模型加载成功！")
        except Exception as e:
            print(f"抬腿模型加载失败: {e}")
        return ep
    else:
        print("未找到已保存的抬腿模型，从头开始训练")
        return default_episode

def load_decision_model(decision_agent, dec_dir: Optional[str]) -> int:
    """可选加载决策模型（若目录存在且有文件）。返回起始决策回合编号。"""
    if not dec_dir:
        return 0
    latest_dec, dec_ep = _latest_decision_ckpt(dec_dir)
    if latest_dec:
        try:
            ckpt = torch.load(latest_dec)
            if isinstance(ckpt, dict) and 'policy' in ckpt:
                decision_agent.policy.load_state_dict(ckpt['policy'])
                if 'optimizer' in ckpt and decision_agent.optimizer:
                    decision_agent.optimizer.load_state_dict(ckpt['optimizer'])
                print(f"决策模型加载成功: {latest_dec}")
            return dec_ep
        except Exception as e:
            print(f"决策模型加载失败: {e}")
    return 0


def PPO_episoid_1(model_path=None, max_steps_per_episode=5):
    # ===== 训练管理器初始化（方案A核心） =====
    training_manager = TrainingManager()
    
    # ===== 智能体实例化（统一放置） =====
    hppo_agent = hppo(num_servos=2, node_num=19, env_information=None)          # 抓取阶段智能体
    tai_agent = hppo(num_servos=3, node_num=19, env_information=None)       # 抬腿阶段智能体（复用）
    decision_hppo_agent = d_hppo(num_servos=1, node_num=19, env_information=None)  # 上层决策智能体

    # ===== 日志写入器 =====
    log_writer_catch = Log_write()  # 创建抓取日志写入器
    log_writer_tai = Log_write()    # 创建抬腿日志写入器
    log_writer_decision = Log_write()  # 创建决策日志写入器

    # ===== 基础计数 =====
    tai_episoid = 1

    # ===== 模型保存目录（统一，使用配置的新路径） =====
    catch_checkpoint_dir = path_list['model_path_catch_PPO_h']
    decision_checkpoint_dir = path_list['model_path_decision_PPO_h']
    _ensure_dir(catch_checkpoint_dir)
    _ensure_dir(decision_checkpoint_dir)

    # ===== 日志文件（自动递增编号） =====
    # 确保日志目录存在
    _ensure_dir(path_list['catch_log_path_PPO'])
    _ensure_dir(path_list['tai_log_path_PPO'])
    _ensure_dir(path_list['decision_log_path_PPO'])
    
    log_file_latest_catch = _next_log_file(path_list['catch_log_path_PPO'], 'catch_log')
    log_file_latest_tai = _next_log_file(path_list['tai_log_path_PPO'], 'tai_log')
    log_file_latest_decision = _next_log_file(path_list['decision_log_path_PPO'], 'decision_log')
    print(f"将使用新的抓取日志目录: {log_file_latest_catch}")
    print(f"将使用新抬腿的日志目录: {log_file_latest_tai}")
    print(f"将使用新决策的日志目录: {log_file_latest_decision}")

    # ===== 模型加载（函数化） =====
    episode_start = load_catch_model(model_path, hppo_agent, path_list['model_path_catch_PPO_h'])

    tai_episoid = load_tai_model(tai_agent, path_list['model_path_tai_PPO_h'], default_episode=tai_episoid)

    decision_episode = load_decision_model(decision_hppo_agent, path_list.get('model_path_decision_PPO_h'))

    # ===== 索引与计数（集中管理） =====
    episode_num = episode_start           # 抓取阶段起始轮次
    # 决策层起始轮次（从决策模型文件名恢复，若不存在则为0）
    decision_episode = decision_episode if 'decision_episode' in locals() else 0
    total_episode = decision_episode      # 总轮次计数
    success_catch = 0                     # 抓取成功次数
    catch_success = False                 # 跨episode标记：上一轮是否抓取成功

    # ===============================
    # 上层总训练循环（新增）
    # ===============================
    MAX_TOTAL_EPISODE = 3000

    env = Environment()  # 仍然只有一个 env

    while total_episode < MAX_TOTAL_EPISODE:

        print(f"\n==============================")
        print(f"🌍 Total Episode {total_episode}")
        print(f"==============================")

        # ---------- 上层决策 ----------
        # 修复：添加 imgs 参数
        d_steps = 0
        d_obs_img, d_obs_tensor = env.get_img(d_steps)
        d_robot_state = env.get_robot_state()
        d_obs = (d_obs_tensor, d_robot_state)
        
        # 调试：打印输入形状和值范围
        print(f"📊 d_obs_tensor shape: {d_obs_tensor.shape}, range: [{d_obs_tensor.min():.3f}, {d_obs_tensor.max():.3f}]")
        print(f"📊 d_robot_state shape: {len(d_robot_state) if isinstance(d_robot_state, list) else d_robot_state.shape}")
        
        decision_dict = decision_hppo_agent.choose_action(
            obs=d_obs,
            x_graph=d_robot_state
        )

        # decision ∈ {0,1}
        decision = int(decision_dict['discrete_action'][0])
        print(f"上层决策 decision = {decision} (0=抓取, 1=爬梯)")
        decision = 0

        if decision == 0:
            print("🟢 进入【抓取训练阶段】")
            # catch_success由上一轮保持，不重置（除非抬腿完成后）
            env.reset()
            env.wait(500)
            steps = 0
            # 直接使用外层的episode_num，不用for循环
            log_writer_catch.add(episode_num=episode_num)
            print(f"<<<<<<<<<第{episode_num}周期")  # 打印当前周期
            
            print("____________________")  # 打印初始状态
           
            while True:
                obs_img, obs_tensor = env.get_img(steps)  # 获取初始图像和图像张量

                robot_state = env.get_robot_state()
                obs = (obs_tensor, robot_state)
                dict = hppo_agent.choose_action(obs=obs,x_graph=robot_state)
                
                d_action = dict['discrete_action']
                c_action = dict['continuous_action']
                action_shoulder = dict['continuous_action'][0]
                action_arm = dict['continuous_action'][1]
                log_prob_shoulder = dict['continuous_log_prob'][0]
                log_prob_arm = dict['continuous_log_prob'][1]
                value = dict['value']

                next_state, reward, done, catch_success = env.step(d_action, c_action, steps)

                next_obs_img, next_obs_tensor = env.get_img(steps)  # 获取下一个图像和图像张量

                hppo_agent.store_transition(
                            state=[obs_img, robot_state, robot_state],
                            discrete_action=d_action,
                            continuous_action=dict['continuous_action'],
                            reward=reward,
                            next_state=[next_obs_img, next_state, next_state],
                            done=done,
                            value=value,
                            discrete_log_prob=dict['discrete_log_prob'],
                            continuous_log_prob=dict['continuous_log_prob']
                        )
                
                loss_d, loss_c = hppo_agent.learn()

                obs_tensor = next_obs_tensor
                robot_state = next_state

                steps += 1
                print("已执行")
                if done == 1 or steps >= max_steps_per_episode:
                    break

                ## 应该还有一个存储模型然后就够了
        else:
            print("🟢 进入【抬腿训练阶段】")    
            # 只有抓取成功后才允许抬腿；未抓取成功则跳过本轮抬腿
            if not catch_success:
                print("⚠️ 未检测到抓取成功，本轮跳过抬腿训练。")
                continue
            # if success_flag1 == 1:
            #     success_catch += 1
            #     log_writer_catch.add(success_catch=success_catch)
            #     print("success_catch:", success_catch)
            #     print("抓取成功，开始抬腿训练...")
            #     total_episode = i
            print("tai_episoid:", tai_episoid)
            PPO_tai_episoid(existing_env=env, total_episode=total_episode, episode=tai_episoid,
                            log_writer_tai=log_writer_tai, log_file_latest_tai=log_file_latest_tai,
                            catch_success=catch_success, tai_agent=tai_agent, training_manager=training_manager)
            tai_episoid += 1
            
            # 抬腿执行完毕，重置环境和抓取标记，准备下一个完整循环
            catch_success = False
            env.reset()
            env.wait(500)

        # ===== 决策奖励计算（基于状态判断是否正确） =====
        decision_reward = 0.0
        if decision == 0:  # 决策选择抓取
            if catch_success:
                # 已经抓取成功还选择抓取，决策错误！
                decision_reward = -15.0
                print("❌ 决策错误：已抓取成功还选择抓取，惩罚-15.0")
            else:
                # 未抓取时选择抓取，决策正确
                decision_reward = 5.0
                print("✅ 决策正确：未抓取状态选择抓取，奖励+5.0")
        else:  # decision == 1，决策选择抬腿
            if catch_success:
                # 已抓取成功且选择抬腿，决策正确
                decision_reward = 10.0
                print("✅ 决策正确：已抓取状态选择抬腿，奖励+10.0")
            else:
                # 未抓取却选择抬腿，决策错误
                decision_reward = -10.0
                print("❌ 决策错误：未抓取状态选择抬腿，惩罚-10.0")
        
        # 记录决策日志
        log_writer_decision.add(episode_num=total_episode)
        log_writer_decision.add(decision=decision)
        log_writer_decision.add(decision_reward=decision_reward)
        log_writer_decision.add(catch_success=int(catch_success))
        
        # 决策智能体需要 state 包含 (x, state, x_graph)，将机器人状态复用为图输入以满足长度要求
        decision_state = (d_obs_tensor, d_robot_state, d_robot_state)
        decision_hppo_agent.store_transition(
            state=decision_state,
            action=decision,
            reward=decision_reward,  # 使用计算得到的奖励
            next_state=None,
            done=True,
            value=decision_dict['value'],
            log_prob=decision_dict['discrete_log_prob']
        )

        # 记录决策的value值
        log_writer_decision.add(decision_value=decision_dict['value'])
        
        # 【方案A】使用训练管理器控制学习频率
        training_manager.increment_decision()
        if training_manager.should_learn_decision():
            decision_loss = decision_hppo_agent.learn()
            print(f'【决策模型学习】{training_manager.get_status()} | decision_loss: {decision_loss:.6f}')
            log_writer_decision.add(decision_loss=decision_loss)
        else:
            # 累积经验但不学习
            print(f'【决策模型累积经验】{training_manager.get_status()}')
            decision_loss = 0
        
        # 保存决策日志
        log_writer_decision.save_catch(log_file_latest_decision)

        # 定期保存决策智能体
        if total_episode % 50 == 0:
            dec_path = os.path.join(decision_checkpoint_dir, f"decision_hppo_{total_episode}.ckpt")
            dec_ckpt = {
                'policy': decision_hppo_agent.policy.state_dict(),
                'optimizer': decision_hppo_agent.optimizer.state_dict(),
                'episode': total_episode
            }
            torch.save(dec_ckpt, dec_path)

        total_episode += 1

    # 如果整个训练过程结束，返回抓取成功状态和环境实例
    return False, env