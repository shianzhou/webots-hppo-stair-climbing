
该文件只存放于webot交互的接口，Darwin类，来管理Robot里面的接口

由于历史原因，里面的存在有一个env类，用于做step，也就是说所有代码**与webots交互的最高层次就是env类**，env.step()用于执行抓取，step1（）用于执行抬腿

所以大致关系就是

```
Darwin类

def robot_reset(self):
```

### 第一改

env->RobotRun->Darwin->webots

奖励函数设计应该放在RobotRun里面
后续需要再给决策层加一step
接口管理

也就是这个是step，在step里面调用RobotRun
```
 def step(self, state, action_shouder, action_arm, steps, catch_flag, gps1, gps2, gps3, gps4, img_name):

        """执行一步动作

        参数:

            state: 当前状态

            action_shouder: 肩膀舵机动作

            action_arm: 手臂舵机动作

            steps: 步数

            catch_flag: 抓取器状态

            gps1-4: GPS位置信息

            name: 动作名称

        返回:

            tuple: (next_state, reward, done, good, goal, count)

        """

        from python_scripts.PPO.RobotRun1 import RobotRun

        return RobotRun(self.robot, state, action_shouder, action_arm, steps, catch_flag, gps1, gps2, gps3, gps4, img_name).run()
```




# 接口介绍

### get_robot_state(self)

得到**所有关节**状态

```
def get_robot_state(self):

        """获取机器人的关节状态，即舵机角度"""

        return self.darwin.get_robot_state()
```

Darwin类里
```
 def get_robot_state(self):

        """获取机器人关节状态
        返回：
            list: 包含所有关节位置的列表
        """

        robot_state = [sensor.getValue() for sensor in self.motors_sensors[:-2]]

        return robot_state
```