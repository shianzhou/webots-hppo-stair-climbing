import os

import torch

from python_scripts.PPO.PPO_episoid_2_1 import PPO_tai_episoid
from python_scripts.PPO.checkpoint_utils import (
    ensure_dir as _ensure_dir,
    load_catch_model,
    load_decision_model,
    load_tai_model,
    next_log_file as _next_log_file,
)
from python_scripts.PPO.training_manager import TrainingManager
from python_scripts.PPO.hppo import HPPO as d_hppo
from python_scripts.PPO.hppo_01 import HPPO as hppo
from python_scripts.PPO_Log_write import Log_write
from python_scripts.Project_config import path_list
from python_scripts.Webots_interfaces import Environment


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
    catch_save_interval = 500
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
            done = 0
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

            if done == 1 and total_episode % catch_save_interval == 0:
                catch_path = os.path.join(catch_checkpoint_dir, f"catch_hppo_{total_episode}.ckpt")
                hppo_agent.save_checkpoint(catch_path, episode=total_episode)
                print(f"保存抓取模型到: {catch_path}")

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