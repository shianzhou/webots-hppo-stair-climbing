RobotRun1
用于做抓取动作，由于没有用到YOLO，首先把Yolo相关的代码全给注释了


在该文件里面实现抓取动作，是不是奖励也应该放在里面呢

robotrun的逻辑应该是得到了由网络得到的下一步的舵机值，然后去将这个舵机值运用上

判断抓取成功的方式是，压力传感器的值是否与目标值相等
神了，去webots上看就是两片传感器，

episode里面catch_flag跟success_flag满天飞，没有一个统一的指标，我都忘了我要干什么了，我是在找哪一步执行的抓取

你切记，要找谁动，就找舵机是谁就行

```
self.motors[21].setPosition(-0.5)  # 电机21设置位置

self.motors[20].setPosition(-0.5)  # 电机20设置位置
```


```
for j in range(len(self.touch)):

self.touch_value[j] = self.touch[j].getValue()  # 压力传感器值

sucess =np.array_equal(self.touch_value,Darwin_config.touch_T)  # 成功标识符=压力传感器值与目标值相等    

sucess = np.int(sucess)  # 成功标识符=1

faild = np.array_equal(self.touch_value,Darwin_config.touch_F)  # 失败标识符=压力传感器值与失败值相等

faild = np.int(faild)  # 失败标识符=1
```


当前 run的return值为
```
return self.next_state, \

self.return_flag_list['reward'], \

self.return_flag_list['done'], \

self.return_flag_list['good'], \

self.return_flag_list['goal'], \

self.return_flag_list['count']
```

而count值根本没有用到

# run的作用

用于执行网络给出的离散值动作和连续动作
根据离散动作判断舵机是否运动

并在每一个run之中计算该步的奖励值，这是每一步的奖励值

episode应该是纯净的，状态的得到都是通过env类来得到，所以在内部，在robotrun里面得到也一样的
# run 的返回值

```
reward':0,（没有用在外面重新赋值）

'done'  :0,是否抓取成功（通过压力传感）

'good'  :0,

'goal'  :0, 是否抓取的是目标梯（GPS）

'count' :0
```

return flag我觉得是要更改的，有些部分是不必要的比如count，ppo的计算需要done，所以done是必须的
根据我的想法，step的返回值应该就是奖励，和状态用于网络的更新

有些部分是不必要的比如count，ppo的计算需要done，所以done是必须的，判断是否抓取成功我在外面有一个catch_success所以返回值应该有他一份，当然返回值也应该有奖励值一份，next——state是用与计算ppo的，但是不是应该由step得到我觉得有待商榷

当前已经修改为

```
return self.next_state, \

self.return_flag_list['reward'], \

self.return_flag_list['done'], \

self.return_flag_list['catch_success']
```

# 奖励构成

- reward1  reward2不应该是奖励或者说定位很奇怪，他应该是判断力抓住的台阶是不是对的，并且作为奖励的的存在

# 重构计划

该文件应该是执行抓取动作并且给抓取奖励