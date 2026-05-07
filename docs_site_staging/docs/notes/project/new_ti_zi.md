该文件也是很神奇了

在webots里面有除了环境外（比如梯子和地板）有一个robot实体和supervisor

supervisor的功能远比robot节点的功能要强大，但具体我也说不太出来

但是记住supervisor可以调用一切节点，所以使用了supervisor对整个机器人的位置进行重置

在运行中 supervisor和robot是可以看作并行的，只是supervisor调用了robot这个处于整个世界里的结点

所以事实上有两个东西在运行，一个是new_ti_zi一个是train_main，两个文件通过resetflag系列文件来进行通信，也就是说两个东西都能改变里面的标志，并利用里面的标志作为判断

所以不用管new_ti_zi，他只是在需要重置机器人位置的时候也就是reset文件被改变并且被检测到的时候启用，**注意如果你要改变训练代码你是否动用了resetflag文件**，

# reset

```
def reset(self):

        print("++++reset++++")

        a = np.random.uniform(0.015, 0.02)

        DAR = [-0.0176538, 0.332399, -0.00606099]   # 机器人三维坐标的默认值

        sui_ji = [np.random.random() * 0.15 - 0.075, 0, np.random.random() * -0.02]   # 随机化机器人在三维世界中的初始位置坐标

        for i in range(0, 3):

            DAR[i] = DAR[i] + sui_ji[i]

        self.robotis_op2_trans.setSFVec3f(DAR)   # 调整机器人在仿真世界中的位置

        self.wait(100)

        self.robotis_op2_rotation.setSFRotation([0.999989, 0.000865437, 0.00465503, 0.296348])   # 调整机器人在仿真世界中的朝向

        self.wait(100)

        self.robotis_op2_trans.setSFVec3f(DAR)   # 调整机器人在仿真世界中的位置

        self.wait(100)

    def resetsimulation(self):

        self.robot.step(self.timestep)

        isremove = True

        while True:

            with open('E:\\project_MultiAgent_h_change\\python_scripts\\PPO\\resetFlag.txt', 'r') as file:

                flag = file.read()

                if flag == '0':

                    with open('E:\\project_MultiAgent_h_change\\python_scripts\\PPO\\resetFlag.txt', 'r+') as file:

                        file.write('1')

                    self.robot.simulationResetPhysics()

                    self.reset()

            self.robot.step(self.timestep)
```
