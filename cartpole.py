import gym
import tensorflow as tf
import numpy as np
import cv2
import random

width = 30
height = 30
num_moves = 18
learning_rate = 0.001
game_name = 'Berzerk-v0'
save_dir = '/checkpoint/'+game_name+'/'

def cnn(X,act):
    X = tf.reshape(X,shape=[-1, width, height, 1])
    c1 = tf.layers.conv2d(X, 16, (5,5),activation=tf.nn.relu)
    c2 = tf.layers.conv2d(c1, 24, (3,3), activation=tf.nn.relu)
    fc = tf.contrib.layers.flatten(c2)
    fc = tf.concat([fc,act],1)  #
    fc = tf.layers.dense(fc,25,activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc,1) # ,activation=tf.nn.softmax
    return fc2

X = tf.placeholder(tf.float32, [None, width, height])
act = tf.placeholder(tf.float32, [None, num_moves])
Y = tf.placeholder(tf.float32)  # (1-a)*Q(s,act)+a*(r+y*maxQ(s',act'))
out_ = cnn(X,act)
#predict = tf.argmax(out_,1)

loss = tf.losses.mean_squared_error(out_,Y)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

def norm(obs):
    if len(obs.shape) == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs,(width,height))
    return obs/255.0

env = gym.make(game_name)    # CartPole-v0, Berzerk-v0, SpaceInvadersDeterministic-v0
jList = []
rList = []
y = 0.99
e = 0.1
alpha = 0.1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #for i_episode in range(20):
    for t in range(2000):
        d = False
        obs = env.reset()
        for j in range(2000):
            env.render()
            obs = norm(obs)
            allQ = sess.run([out_],feed_dict={X:[obs for i in range(num_moves)],act:np.identity(num_moves)}) #
            allQ = np.transpose(allQ)[0]
            a = np.argmax(allQ) #
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            obs1,r,d,_ = env.step(a)
            print(a,r,allQ[a][0])
            obs1 = norm(obs1)
            maxQ = sess.run([out_],feed_dict={X:[obs1 for i in range(num_moves)],act:np.identity(num_moves)})
            maxQ = np.max(maxQ)
            y = (1.0-alpha)*allQ[a][0] + alpha*(r+y*maxQ)
            sess.run(train,feed_dict={X:[obs],act:[np.identity(num_moves)[a]],Y:[y]})
            obs = obs1
            if d == True:
                break

