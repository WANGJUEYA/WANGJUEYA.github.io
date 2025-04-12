import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# gym=0.26.0 https://blog.csdn.net/qq_43674552/article/details/127344366

# Hyper Parameters 超参数
EPOCH = 400  # 400个episode循环
BATCH_SIZE = 32  # 样本数量
LR = 0.01  # learning rate | 学习率
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency | 目标网络更新频率
MEMORY_CAPACITY = 2000  # 记忆库容量
env = gym.make('CartPole-v0')  # 使用gym库中的环境：CartPole，且打开封装
env = env.unwrapped  # 打开环境封装
N_ACTIONS = env.action_space.n  # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]  # 杆子状态个数 (4个)

"""
torch.nn是专门为神经网络设计的模块化接口。nn构建于Autograd之上，可以用来定义和运行神经网络。
nn.Module是nn中十分重要的类，包含网络各层的定义及forward方法。
定义网络：
    需要继承nn.Module类，并实现forward方法。
    一般把网络中具有可学习参数的层放在构造函数__init__()中。
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
"""


# 定义Net类 (定义网络)
class Net(nn.Module):

    def __init__(self):  # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()  # 等价与nn.Module.__init__()
        self.fc1 = nn.Linear(N_STATES, 20)  # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到20个神经元
        self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 = nn.Linear(20, N_ACTIONS)  # 设置第二个全连接层(隐藏层到输出层): 20个神经元到动作数个神经元
        self.fc2.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):  # 定义forward函数 (x为状态)
        x = F.relu(self.fc1(x))  # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        return self.fc2(x)  # 连接隐藏层到输出层，获得最终的输出值 (即动作值)


# 定义DQN类 (定义两个网络)
class DQN(object):

    def __init__(self):  # 定义DQN的一系列属性
        self.target_net, self.evaluate_net = Net(), Net()  # 利用Net创建两个神经网络: 评估网络和目标网络
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库，一行代表一个transition
        self.loss_Function = nn.MSELoss()  # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=LR)  # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.point = 0  # for storing memory
        self.learn_step = 0  # for target updating

    def choose_action(self, s):  # 定义动作选择函数 (s为状态)
        s = torch.unsqueeze(torch.FloatTensor(s), 0)  # 将s转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON:  # epsilon-greedy 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            return torch.max(self.evaluate_net.forward(s), 1)[1].data.numpy()[0]  # 通过对评估网络输入状态s，前向传播获得动作值
        else:  # 随机选择动作
            return np.random.randint(0, N_ACTIONS)  # 这里action随机等于0或1 (N_ACTIONS = 2)

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        self.memory[self.point % MEMORY_CAPACITY, :] = np.hstack((s, [a, r], s_))  # 如果记忆库满了，便覆盖旧的数据
        self.point += 1  # memory_counter自加1

    def sample_batch_data(self, batch_size):  # 抽取记忆库中的批数据
        perm_idx = np.random.choice(len(self.memory), batch_size)
        return self.memory[perm_idx]

    def learn(self) -> float:  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step % TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.evaluate_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step += 1  # 学习步数自加1

        # 抽取32个索引对应的32个transition，存入batch_memory
        batch_memory = self.sample_batch_data(BATCH_SIZE)
        # 将32个s抽出，转为32-bit floating point形式，并存储到batch_state中，batch_state为32行4列
        batch_state = torch.FloatTensor(batch_memory[:, :N_STATES])
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到batch_action中 (LongTensor类型方便后面torch.gather的使用)，batch_action为32行1列
        batch_action = torch.LongTensor(batch_memory[:, N_STATES: N_STATES + 1].astype(int))
        # 将32个r抽出，转为32-bit floating point形式，并存储到batch_reward中，batch_reward为32行1列
        batch_reward = torch.FloatTensor(batch_memory[:, N_STATES + 1: N_STATES + 2])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到batch_next_state中，batch_next_state为32行4列
        batch_next_state = torch.FloatTensor(batch_memory[:, -N_STATES:])

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.evaluate_net(batch_state).gather(1, batch_action)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(batch_next_state).detach()  # target network
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_Function(q_eval, q_target)

        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数

        return loss.data.numpy()  # 返回损失函数数值


if __name__ == "__main__":
    dqn = DQN()

    writer = SummaryWriter("run/MemoryCapacity_100_CustomReward/")
    writer.add_graph(dqn.evaluate_net, torch.randn(1, N_STATES))

    global_step = 0  # 绘图横坐标
    for i in range(EPOCH):  # episode循环
        s = env.reset()  # 重置环境
        running_loss = 0  # 损失函数值
        cumulated_reward = 0  # 初始化该循环对应的episode的总奖励
        step = 0

        while True:
            global_step += 1
            env.render()  # 显示实验动画
            a = dqn.choose_action(s)  # 输入该步对应的状态s，选择动作
            s_, r, done, _ = env.step(a)  # 执行动作，获得反馈

            # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(s, a, r, s_)  # 存储样本

            cumulated_reward += r  # 逐步加上一个episode内每个step的reward
            if dqn.point > MEMORY_CAPACITY:  # 如果累计的transition数量超过了记忆库的固定容量2000
                # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
                loss = dqn.learn()
                running_loss += loss
                if done or step > 2000:
                    print("��FAIL��Episode: %d| Step: %d| Loss:  %.4f, Reward: %.2f" % (
                        i, step, running_loss / step, cumulated_reward))
                    writer.add_scalar("training/Loss", running_loss / step, global_step)
                    writer.add_scalar("training/Reward", cumulated_reward, global_step)
                    break
            else:
                print("\rCollecting experience: %d / %d..." % (dqn.point, MEMORY_CAPACITY), end='')

            if done:
                break
            if step % 100 == 99:
                print("Episode: %d| Step: %d| Loss:  %.4f, Reward: %.2f" % (
                    i, step, running_loss / step, cumulated_reward))
            step += 1
            s = s_
