import os

from python_scripts.PPO.PPO_episoid_2_1 import PPO_tai_episoid
from python_scripts.PPO.preparation_tool.checkpoint_utils import (
    ensure_dir as _ensure_dir,
    load_catch_model,
    load_decision_model,
    load_tai_model,
    next_log_file as _next_log_file,
)
from python_scripts.PPO.RobotRun0 import RobotRun0
from python_scripts.PPO.preparation_tool.training_manager import TrainingManager
from python_scripts.PPO.hppo import HPPO as d_hppo
from python_scripts.PPO.hppo_01 import HPPO as hppo
from python_scripts.PPO.log_code import CatchAgentLog, TaiAgentLog, DecisionAgentLog
from python_scripts.Project_config import path_list
from python_scripts.Webots_interfaces import Environment


def init_training_and_logging(path_list, default_tai_episode=1, catch_save_interval=500):
    """
    Initialize training manager, agents, log writers, directories and return them as a dict.
    """
    training_manager = TrainingManager()

    # 智能体实例化（统一放置）
    hppo_agent = hppo(num_servos=2, node_num=19, env_information=None)
    tai_agent = hppo(num_servos=3, node_num=19, env_information=None)
    decision_hppo_agent = d_hppo(num_servos=1, node_num=19, env_information=None)

    # 日志写入器（使用新的log_code包）
    log_writer_catch = CatchAgentLog(keep_records=False)
    log_writer_tai = TaiAgentLog(keep_records=False)
    log_writer_decision = DecisionAgentLog(keep_records=False)

    tai_episoid = default_tai_episode

    # 模型保存目录（统一，使用配置的新路径）
    catch_checkpoint_dir = path_list['model_path_catch_PPO_h']
    decision_checkpoint_dir = path_list['model_path_decision_PPO_h']

    _ensure_dir(catch_checkpoint_dir)
    _ensure_dir(decision_checkpoint_dir)

    # 日志目录（确保存在）
    _ensure_dir(path_list['catch_log_path_PPO'])
    _ensure_dir(path_list['tai_log_path_PPO'])
    _ensure_dir(path_list['decision_log_path_PPO'])

    log_file_latest_catch = _next_log_file(path_list['catch_log_path_PPO'], 'catch_log')
    log_file_latest_tai = _next_log_file(path_list['tai_log_path_PPO'], 'tai_log')
    log_file_latest_decision = _next_log_file(path_list['decision_log_path_PPO'], 'decision_log')

    print(f"将使用新的抓取日志目录: {log_file_latest_catch}")
    print(f"将使用新抬腿的日志目录: {log_file_latest_tai}")
    print(f"将使用新决策的日志目录: {log_file_latest_decision}")

    return {
        'training_manager': training_manager,
        'hppo_agent': hppo_agent,
        'tai_agent': tai_agent,
        'decision_hppo_agent': decision_hppo_agent,
        'log_writer_catch': log_writer_catch,
        'log_writer_tai': log_writer_tai,
        'log_writer_decision': log_writer_decision,
        'tai_episoid': tai_episoid,
        'catch_checkpoint_dir': catch_checkpoint_dir,
        'decision_checkpoint_dir': decision_checkpoint_dir,
        'catch_save_interval': catch_save_interval,
        'log_file_latest_catch': log_file_latest_catch,
        'log_file_latest_tai': log_file_latest_tai,
        'log_file_latest_decision': log_file_latest_decision,
    }


def PPO_episoid_1(model_path=None, max_steps_per_episode=5):
    # 初始化训练管理与日志（已提取）
    init = init_training_and_logging(path_list)

    training_manager = init['training_manager']
    hppo_agent = init['hppo_agent']
    tai_agent = init['tai_agent']
    decision_hppo_agent = init['decision_hppo_agent']

    log_writer_catch = init['log_writer_catch']
    log_writer_tai = init['log_writer_tai']
    log_writer_decision = init['log_writer_decision']

    tai_episoid = init['tai_episoid']

    catch_checkpoint_dir = init['catch_checkpoint_dir']
    decision_checkpoint_dir = init['decision_checkpoint_dir']
    catch_save_interval = init['catch_save_interval']

    log_file_latest_catch = init['log_file_latest_catch']
    log_file_latest_tai = init['log_file_latest_tai']
    log_file_latest_decision = init['log_file_latest_decision']

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
    MAX_TOTAL_EPISODE = 10000

    env = Environment()  # 仍然只有一个 env
    robot_run0 = RobotRun0(
        decision_agent=decision_hppo_agent,
        training_manager=training_manager,
        log_writer_decision=log_writer_decision,
        log_file_latest_decision=log_file_latest_decision,
        decision_checkpoint_dir=decision_checkpoint_dir,
    )

    while total_episode < MAX_TOTAL_EPISODE:

        print(f"\n==============================")
        print(f"🌍 Total Episode {total_episode}")
        print(f"==============================")

        # ---------- 上层决策 ----------
        d_steps = 0
        d_obs_img, d_obs_tensor = env.get_img(d_steps)
        d_robot_state = env.get_robot_state()
        decision_dict, decision, decision_state = robot_run0.choose_action(d_obs_tensor, d_robot_state)
        pre_branch_catch_success = catch_success
        route_info = robot_run0.judge_route(decision=decision, catch_success=pre_branch_catch_success)
        route = route_info['route']

        if route == 'catch':
            print("🟢 进入【抓取训练阶段】")
            # catch_success由上一轮保持，不重置（除非抬腿完成后）
            env.reset()
            env.wait(500)
            steps = 0
            done = 0
            episode_reward = 0  # 累计episode奖励
            loss_d = 0.0  # 最后一次的离散损失
            loss_c = 0.0  # 最后一次的连续损失
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
                episode_reward += reward  # 累计奖励

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
                
                loss_d, loss_c = hppo_agent.learn()  # 保存最后一次loss

                obs_tensor = next_obs_tensor
                robot_state = next_state

                steps += 1
                print("已执行")
                if done == 1 or steps >= max_steps_per_episode:
                    break

            # 记录catch episode日志
            log_writer_catch.add_episode(
                episode_num=episode_num,
                total_episode=total_episode,
                loss_discrete=loss_d,
                loss_continuous=loss_c,
                episode_reward=episode_reward,
                episode_steps=steps,
                catch_success=catch_success
            )
            log_writer_catch.save(log_file_latest_catch)

            if done == 1 and total_episode % catch_save_interval == 0:
                catch_path = os.path.join(catch_checkpoint_dir, f"catch_hppo_{total_episode}.ckpt")
                hppo_agent.save_checkpoint(catch_path, episode=total_episode)
                print(f"保存抓取模型到: {catch_path}")

                ## 应该还有一个存储模型然后就够了
        elif route == 'tai':
            print("🟢 进入【抬腿训练阶段】")    
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
        else:
            print("🟡 决策不满足执行条件，本轮不进入抓取/抬腿，直接重新决策。")

        robot_run0.finalize(
            total_episode=total_episode,
            decision=decision,
            decision_dict=decision_dict,
            decision_state=decision_state,
            route=route,
            pre_branch_catch_success=pre_branch_catch_success,
            post_branch_catch_success=catch_success,
        )

        total_episode += 1

    # 如果整个训练过程结束，返回抓取成功状态和环境实例
    return False, env