import numpy as np
import json
import re
from datetime import datetime


class CustomJSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 NumPy 数组、PyTorch 张量和 datetime 对象"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif "torch.Tensor" in str(type(obj)):
            try:
                return obj.cpu().detach().numpy().tolist()
            except (AttributeError, Exception):
                pass
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return super().default(obj)


class BaseLogWriter:
    """通用日志写入器基类 - 支持 series 和 records 两种记录方式"""
    def __init__(self, keep_records=True):
        self.keep_records = bool(keep_records)
        self.data = {
            'start time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'save time': [],
            'series': {},  # 按列组织的数据 {key: [v1, v2, ...]}
        }
        if self.keep_records:
            self.data['records'] = []  # 逐条记录 [{k1: v1, k2: v2}, ...]

    def _normalize_scalar(self, value):
        """将张量、NumPy 值转换为 Python 原生类型"""
        if hasattr(value, 'item') and callable(value.item):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, np.ndarray) and value.size == 1:
            return value.item()
        return value

    def _append_series_record(self, record):
        """追加记录到 series（列式存储）"""
        series = self.data.setdefault('series', {})
        existing_count = 0
        if series:
            first_key = next(iter(series))
            existing_count = len(series[first_key])

        for key in record:
            if key not in series:
                series[key] = [None] * existing_count

        for key, values in series.items():
            values.append(self._normalize_scalar(record.get(key)))

    def _is_scalar(self, value):
        """判断是否为标量"""
        return value is None or isinstance(value, (bool, int, float, str))

    def _dump_with_inline_lists(self, obj, indent=4, level=0):
        """格式化 JSON（数组行内显示）"""
        pad = ' ' * (indent * level)
        next_pad = ' ' * (indent * (level + 1))

        if isinstance(obj, dict):
            if not obj:
                return '{}'
            items = []
            for key, value in obj.items():
                key_text = json.dumps(str(key), ensure_ascii=False)
                value_text = self._dump_with_inline_lists(value, indent, level + 1)
                items.append(f"{next_pad}{key_text}: {value_text}")
            return '{\n' + ',\n'.join(items) + '\n' + pad + '}'

        if isinstance(obj, list):
            if not obj:
                return '[]'
            if all(self._is_scalar(v) for v in obj):
                return json.dumps(obj, ensure_ascii=False, separators=(', ', ': '))
            items = [self._dump_with_inline_lists(v, indent, level + 1) for v in obj]
            return '[\n' + ',\n'.join(f"{next_pad}{item}" for item in items) + '\n' + pad + ']'

        return json.dumps(obj, ensure_ascii=False)

    def add_cycle_record(self, **kwargs):
        """添加一条循环记录"""
        record = {key: self._normalize_scalar(value) for key, value in kwargs.items()}
        if self.keep_records:
            self.data['records'].append(record)
        self._append_series_record(record)

    def add(self, **kwargs):
        """灵活添加字段到最后一条记录"""
        if not kwargs:
            return
        if not self.keep_records:
            self.add_cycle_record(**kwargs)
            return
        if not self.data['records']:
            self.data['records'].append({})
        record = self.data['records'][-1]
        for key, value in kwargs.items():
            record[key] = self._normalize_scalar(value)

    def save(self, file_path):
        """保存日志到文件"""
        self.data['save time'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(f'保存日志到 {file_path}...')

        try:
            data_to_save = json.loads(json.dumps(self.data, cls=CustomJSONEncoder))
        except Exception as e:
            print(f'Error: {e}')
            data_to_save = self.data

        try:
            formatted_json = self._dump_with_inline_lists(data_to_save, indent=4, level=0)
        except Exception as e:
            print(f'Error: {e}')
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            print('✓ 日志保存成功')
        except Exception as e:
            print(f'保存失败: {e}')

    def reset(self):
        """重置日志"""
        self.__init__(keep_records=self.keep_records)

    def get(self, key):
        """获取日志数据"""
        return self.data.get(key, [])

    def clear(self):
        """清空日志"""
        pass


class CatchAgentLogWriter(BaseLogWriter):
    """抓取智能体日志写入器"""
    def __init__(self, keep_records=False):
        super().__init__(keep_records=keep_records)
        self.agent_name = 'CatchAgent'

    def add_episode(self, episode_num=None, total_episode=None, loss_discrete=None, 
                   loss_continuous=None, episode_reward=None, episode_steps=None, 
                   catch_success=None, **extra):
        """记录抓取 episode"""
        record = {
            'episode_num': self._normalize_scalar(episode_num),
            'total_episode': self._normalize_scalar(total_episode),
            'loss_discrete': self._normalize_scalar(loss_discrete),
            'loss_continuous': self._normalize_scalar(loss_continuous),
            'episode_reward': self._normalize_scalar(episode_reward),
            'episode_steps': self._normalize_scalar(episode_steps),
            'catch_success': catch_success,
        }
        for key, value in extra.items():
            record[key] = self._normalize_scalar(value)
        self.add_cycle_record(**record)

    def log_episode(self, file_path, **kwargs):
        """记录 episode 并保存"""
        self.add_episode(**kwargs)
        self.save(file_path)


class TaiAgentLogWriter(BaseLogWriter):
    """抬腿智能体日志写入器"""
    def __init__(self, keep_records=False):
        super().__init__(keep_records=keep_records)
        self.agent_name = 'TaiAgent'

    def add_episode(self, episode_num=None, total_episode=None, loss_discrete=None, 
                   loss_continuous=None, episode_reward=None, episode_steps=None, 
                   tai_success=None, **extra):
        """记录抬腿 episode"""
        record = {
            'episode_num': self._normalize_scalar(episode_num),
            'total_episode': self._normalize_scalar(total_episode),
            'loss_discrete': self._normalize_scalar(loss_discrete),
            'loss_continuous': self._normalize_scalar(loss_continuous),
            'episode_reward': self._normalize_scalar(episode_reward),
            'episode_steps': self._normalize_scalar(episode_steps),
            'tai_success': tai_success,
        }
        for key, value in extra.items():
            record[key] = self._normalize_scalar(value)
        self.add_cycle_record(**record)

    def log_episode(self, file_path, **kwargs):
        """记录 episode 并保存"""
        self.add_episode(**kwargs)
        self.save(file_path)


class DecisionAgentLogWriter(BaseLogWriter):
    """决策智能体日志写入器"""
    def __init__(self, keep_records=False):
        super().__init__(keep_records=keep_records)
        self.agent_name = 'DecisionAgent'

    def add_cycle(self, total_episode=None, decision_action=None, loss_discrete=None, 
                 loss_continuous=None, decision_reward=None, route=None, 
                 pre_catch_success=None, post_catch_success=None, **extra):
        """记录决策循环"""
        record = {
            'total_episode': self._normalize_scalar(total_episode),
            'decision_action': self._normalize_scalar(decision_action),
            'loss_discrete': self._normalize_scalar(loss_discrete),
            'loss_continuous': self._normalize_scalar(loss_continuous),
            'decision_reward': self._normalize_scalar(decision_reward),
            'route': route,
            'pre_catch_success': pre_catch_success,
            'post_catch_success': post_catch_success,
        }
        for key, value in extra.items():
            record[key] = self._normalize_scalar(value)
        self.add_cycle_record(**record)

    def log_cycle(self, file_path, **kwargs):
        """记录循环并保存"""
        self.add_cycle(**kwargs)
        self.save(file_path)


# 向后兼容
class Log_write(BaseLogWriter):
    """通用日志写入器（向后兼容）"""
    def __init__(self, keep_records=True):
        super().__init__(keep_records=keep_records)
