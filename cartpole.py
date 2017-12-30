import gym
import tensorflow as tf
import numpy as np
import cv2
import random
import os
import pylab as pl

height = 21
width = 16
num_moves = 6
num_obs = 2
learning_rate = 0.01
# CartPole-v0, Berzerk-v0, SpaceInvadersDeterministic-v0
game_name = 'Pong-v0'
model_fn = '/checkpoint/'+game_name+'/'+game_name+'.ckpt'

def cnn(X,act):
    X = tf.reshape(X,shape=[-1, num_obs, height, width, 1])
    c1 = tf.layers.conv3d(X, 16, (2,5,5),activation=tf.nn.relu)
    c2 = tf.layers.conv3d(c1, 32, (1,3,3), activation=tf.nn.relu)
    fc = tf.contrib.layers.flatten(c2)
    fc = tf.concat([fc,act],1)  #
    fc = tf.layers.dense(fc,50,activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc,1) # ,activation=tf.nn.softmax
    return fc2

X = tf.placeholder(tf.float32, [None, num_obs, height, width])
act = tf.placeholder(tf.float32, [None, num_moves])
Y = tf.placeholder(tf.float32)  # (1-a)*Q(s,act)+a*(r+y*maxQ(s',act'))
out_ = cnn(X,act)
#predict = tf.argmax(out_,1)

loss = tf.losses.mean_squared_error(out_,Y)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

def norm(obs):
    if len(obs.shape) == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs,(width,height),interpolation=cv2.INTER_AREA) #
    _,obs = cv2.threshold(obs,88,255,cv2.THRESH_BINARY)
    #plot_(obs)
    return obs/255.0

def get_state(env,a):
    res = [env.step(a) for i in range(num_obs)]
    state = [s[0] for s in res]
    r = [s[1] for s in res]
    d = [s[2] for s in res]
    done = False
    for v in d:
        done = done or v
    r = np.sum(r)
    #if done:
    #    r = -50.0
    #s = state[0]
    #for i in range(1,len(state)):
    #    s = np.add(s,state[i])
    #s /= len(state)
    state = [norm(ob) for ob in state]
    return state,r,done

def plot_(img):
    pl.imshow(img) # ,cmap=pl.cm.binary
    pl.pause(.001)
    pl.draw()

def norm_p(v):
    m = np.min(v)
    norm = [a-m for a in v]
    return norm/np.sum(norm)

env = gym.make(game_name)
jList = []
rList = []
y = 0.99
e = 0.1
alpha = 1.0
with tf.Session() as sess:
    saver = tf.train.Saver()
    if os.path.isfile(model_fn+'.meta'):
        saver.restore(sess,model_fn)
    else:
        sess.run(tf.global_variables_initializer())
    for t in range(5000):
        done = False
        obs = env.reset()
        state, r, done = get_state(env,0)
        for j in range(200000):
            if j%10:
                env.render()
            #if j%10:
            #    plot_(state)
            allQ = sess.run([out_],feed_dict={X:[state for i in range(num_moves)],act:np.identity(num_moves)}) #
            allQ = np.transpose(allQ[0])[0]
            a = np.argmax(allQ)
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            state1,r,done = get_state(env,a)
            print(a,r,allQ[a])
            maxQ = sess.run([out_],feed_dict={X:[state1 for i in range(num_moves)],act:np.identity(num_moves)})
            maxQ = np.max(maxQ)
            y = (1.0-alpha)*allQ[a] + alpha*(r+y*maxQ)
            if done:
                y = r
            sess.run(train,feed_dict={X:[state1],act:[np.identity(num_moves)[a]],Y:[y]})
            state = state1
            if done:
                break
            if j%100 == 99:
                saver.save(sess,model_fn)

