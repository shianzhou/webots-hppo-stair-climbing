import glob
import os
import re
from typing import Optional

import torch


def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def next_log_file(dir_path: str, prefix: str) -> str:
    pattern = os.path.join(dir_path, f"{prefix}_*.json")
    existing = glob.glob(pattern)
    max_n = 0
    for file_path in existing:
        match = re.search(rf"{re.escape(prefix)}_(\d+)\.json$", os.path.basename(file_path))
        if match:
            try:
                number = int(match.group(1))
                if number > max_n:
                    max_n = number
            except Exception:
                continue
    return os.path.join(dir_path, f"{prefix}_{max_n + 1}.json")


def latest_catch_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "catch_hppo_*.ckpt"))
    if not files:
        return None, 0

    def _num(file_path: str) -> int:
        match = re.search(r"catch_hppo_(\d+)\.ckpt$", os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    selected = max(files, key=_num)
    return selected, _num(selected)


def latest_tai_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "tai_agent_*_*.ckpt"))
    if not files:
        return None, 0, 0

    def _nums(file_path: str):
        base = os.path.basename(file_path).replace(".ckpt", "")
        parts = base.split("_")
        try:
            return int(parts[-2]), int(parts[-1])
        except Exception:
            return (0, 0)

    selected = max(files, key=_nums)
    total, episode = _nums(selected)
    return selected, total, episode


def latest_decision_ckpt(dir_path: str):
    files = glob.glob(os.path.join(dir_path, "decision_hppo_*.ckpt"))
    if not files:
        return None, 0

    def _num(file_path: str) -> int:
        match = re.search(r"decision_hppo_(\d+)\.ckpt$", os.path.basename(file_path))
        return int(match.group(1)) if match else -1

    selected = max(files, key=_num)
    return selected, _num(selected)


def load_catch_model(model_path: str, hppo_agent, catch_dir: str) -> int:
    episode_start = 0
    if model_path:
        try:
            ckpt = torch.load(model_path)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                hppo_agent.policy.load_state_dict(ckpt["policy"])
                if "optimizer_hppo" in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt["optimizer_hppo"])
                print(f"从指定模型加载: {model_path}，模型加载成功！")
                try:
                    episode_start = int(os.path.basename(model_path).split("_")[-1].split(".")[0])
                    print(f"从指定模型加载: {model_path}，从周期 {episode_start} 继续训练")
                except Exception:
                    pass
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print(f"从指定模型加载: {model_path}，模型加载成功！(旧格式)")
        except Exception as exc:
            print(f"指定模型加载失败: {exc}")
            episode_start = 0
        return episode_start

    selected_model, episode_start = latest_catch_ckpt(catch_dir)
    if selected_model:
        try:
            ckpt = torch.load(selected_model)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                hppo_agent.policy.load_state_dict(ckpt["policy"])
                if "optimizer_hppo" in ckpt and hppo_agent.optimizer:
                    hppo_agent.optimizer.load_state_dict(ckpt["optimizer_hppo"])
                print("抓取模型加载成功！")
            else:
                hppo_agent.policy.load_state_dict(ckpt)
                print("抓取模型加载成功！(旧格式)")
        except Exception as exc:
            print(f"抓取模型加载失败: {exc}")
            episode_start = 0
    else:
        print("未找到已保存的抓取模型，从头开始训练")
        episode_start = 0
    return episode_start


def load_tai_model(tai_agent, tai_dir: str, default_episode: int = 1) -> int:
    selected_tai, _, episode = latest_tai_ckpt(tai_dir)
    if selected_tai:
        print(f"找到最新抬腿模型: {selected_tai}，抬腿周期: {episode}")
        try:
            ckpt = torch.load(selected_tai)
            if isinstance(ckpt, dict) and "policy_tai" in ckpt:
                tai_agent.policy.load_state_dict(ckpt["policy_tai"])
                if "optimizer_tai" in ckpt and tai_agent.optimizer:
                    tai_agent.optimizer.load_state_dict(ckpt["optimizer_tai"])
            print("抬腿模型加载成功！")
        except Exception as exc:
            print(f"抬腿模型加载失败: {exc}")
        return episode
    print("未找到已保存的抬腿模型，从头开始训练")
    return default_episode


def load_decision_model(decision_agent, dec_dir: Optional[str]) -> int:
    if not dec_dir:
        return 0
    latest_dec, dec_episode = latest_decision_ckpt(dec_dir)
    if latest_dec:
        try:
            ckpt = torch.load(latest_dec)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                decision_agent.policy.load_state_dict(ckpt["policy"])
                if "optimizer" in ckpt and decision_agent.optimizer:
                    decision_agent.optimizer.load_state_dict(ckpt["optimizer"])
                print(f"决策模型加载成功: {latest_dec}")
            return dec_episode
        except Exception as exc:
            print(f"决策模型加载失败: {exc}")
    return 0
