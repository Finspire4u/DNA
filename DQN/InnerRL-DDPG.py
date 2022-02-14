import numpy as np
import pandas as pd
import pyglet
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

class antennaEnv(object):
    viewer = None
    dt_r = .1    # refresh rate
    dt_l = .3
    action_bound = [-1, 1]
    goal = {'x': 75., 'y': 75., 'l': 4}
    state_dim = 5
    action_dim = 2

    def __init__(self):
        self.antenna_info = np.zeros(
            2, dtype=[('l', np.float32),('r', np.float32)])
        self.antenna_info['l'] = 25*np.sqrt(2)     # antenna length
        self.antenna_info['r'] = np.pi/6    # angles information
        self.on_goal = 0

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.antenna_info['r'] += action * self.dt_r
        self.antenna_info['r'] %= np.pi * 2    # normalize
        self.antenna_info['l'] += action * self.dt_l

        a1l = self.antenna_info['l'][1]  # radius, antenna length
        a1r = self.antenna_info['r'][0]  # radian, angle
        a1xy = np.array([50., 50.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end (x1, y1)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 100, (self.goal['y'] - a1xy_[1]) / 100]
        r = -np.sqrt(dist1[0]**2+dist1[1]**2)
        # done and reward
        if self.goal['x'] - self.goal['l']/2 < a1xy_[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < a1xy_[1] < self.goal['y'] + self.goal['l']/2:
                r += 2.
                self.on_goal += 1
                if self.on_goal > 10:
                    done = True
        else:
            self.on_goal = 0
        # state
        s = np.concatenate((a1xy_/50, dist1, [1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):
        #Random goal##########################
        self.goal['x'] = np.random.rand()*50. + 50.
        self.goal['y'] = np.random.rand()*50. + 50.
        # Random initial direction############
        self.antenna_info['l'] = 25*np.sqrt(2)    # antenna length
        self.antenna_info['r'] = 0.5 * np.pi * np.random.rand(1)

        # #Random omni angle goal##########################
        # self.goal['x'] = np.random.rand()*100.
        # self.goal['y'] = np.random.rand()*100.
        # # Random omni angle initial direction############
        # self.antenna_info['r'] = 2 * np.pi * np.random.rand(1)

        self.on_goal = 0
        a1l = self.antenna_info['l'][0]  # radius, antenna length
        a1r = self.antenna_info['r'][0]  # radian, angle
        a1xy = np.array([50., 50.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end (x1, y1)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/100, (self.goal['y'] - a1xy_[1])/100]
        # state
        s = np.concatenate((a1xy_/50, dist1, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.antenna_info, self.goal)
        self.viewer.render()
        

class Viewer(pyglet.window.Window):
    bar_thc = 1

    def __init__(self, antenna_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=150, height=150, resizable=False, caption='antenna', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.antenna_info = antenna_info
        self.goal_info = goal
        self.center_coord = np.array([50, 50])
        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.antenna = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [50, 50.5,
                     100, 50.5,
                     100, 49.5,
                     50, 49.5]),
            ('c3B', (249, 86, 86) * 4,))    # color


    def render(self):
        self._update_antenna()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_antenna(self):
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)
        a1l = self.antenna_info['l'][1]     # radius, antenna length
        a1r = self.antenna_info['r'][0]     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end at (x1, y1)
        a1tr = np.pi / 2 - self.antenna_info['r'][0]
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        self.antenna.vertices = np.concatenate((xy01, xy02, xy11, xy12))


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[None, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './DDPGparams', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './DDPGparams')


LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
MAX_EPISODES = 1000
MAX_EP_STEPS = 300
MAX_TEST = 100
MAX_TEST_STEPS = 300
ON_TRAIN = False

# set env
env = antennaEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    ep_r_his = []
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl.choose_action(s)
            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
        ep_r_his.append(ep_r)

    rl.save()
    output = pd.DataFrame(ep_r_his)
    output.to_csv('InnerRL_output2.csv')


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    res = 0
    J = 0
    for i in range(MAX_TEST):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_TEST_STEPS):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            ep_r += r
            if done:
                res += 1
            if done or j == MAX_TEST_STEPS-1:    
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                if J != MAX_TEST_STEPS-1:
                    J+=j
                break
    print('Testing over')
    return [res,J]


if ON_TRAIN:
    train()
else:
    [output,j] = eval()
    print('accuracy:', 100*output/MAX_TEST,'%','average step:', j/output)
