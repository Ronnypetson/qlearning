import gym
import tensorflow as tf
import numpy as np
import cv2
import random
import os

height = 30
width = 90
num_moves = 6
num_obs = 4
learning_rate = 0.001
game_name = 'SpaceInvadersDeterministic-v0' # 'SuperMarioBros-1-1-v0'
model_fn = '/checkpoint/'+game_name+'/'+game_name+'.ckpt'

def cnn(X,act):
    X = tf.reshape(X,shape=[-1, height, width, 1])
    c1 = tf.layers.conv2d(X, 8, (5,5),activation=tf.nn.relu)
    c2 = tf.layers.conv2d(c1, 16, (3,3), activation=tf.nn.relu)
    fc = tf.contrib.layers.flatten(c2)
    fc = tf.concat([fc,act],1)  #
    fc = tf.layers.dense(fc,50,activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc,1) # ,activation=tf.nn.softmax
    return fc2

X = tf.placeholder(tf.float32, [None, height, width])
act = tf.placeholder(tf.float32, [None, num_moves])
Y = tf.placeholder(tf.float32)  # (1-a)*Q(s,act)+a*(r+y*maxQ(s',act'))
out_ = cnn(X,act)
#predict = tf.argmax(out_,1)

loss = tf.losses.mean_squared_error(out_,Y)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

def norm(obs):
    if len(obs.shape) == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs,(width,height)) #
    #plt_(obs)
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
    if done:
        r = -300.0
    s = state[0]
    for i in range(1,len(state)):
        s = np.add(s,state[i])
    s = norm(s)
    return s,r,done

def norm_p(v):
    m = np.min(v)
    v = [v_+m for v_ in v]
    v = v/(np.sum(v))

env = gym.make(game_name)    # CartPole-v0, Berzerk-v0, SpaceInvadersDeterministic-v0
jList = []
rList = []
y = 0.99
e = 0.1
alpha = 0.1
with tf.Session() as sess:
    saver = tf.train.Saver()
    if os.path.isfile(model_fn+'.meta'):
        saver.restore(sess,model_fn)
    else:
        sess.run(tf.global_variables_initializer())
    for t in range(200):
        done = False
        obs = env.reset()
        state, r, done = get_state(env,0)
        for j in range(200000):
            env.render()
            allQ = sess.run([out_],feed_dict={X:[state for i in range(num_moves)],act:np.identity(num_moves)}) #
            allQ = np.transpose(allQ[0])[0]
            #a = np.argmax(allQ) #
            a = np.random.choice(range(num_moves),1,p=norm_p(allQ))[0]
            #if np.random.rand(1) < e:
            #    a = env.action_space.sample()
            state1,r,done = get_state(env,a)
            print(a,r,allQ[a])
            maxQ = sess.run([out_],feed_dict={X:[state1 for i in range(num_moves)],act:np.identity(num_moves)})
            maxQ = np.max(maxQ)
            y = (1.0-alpha)*allQ[a] + alpha*(r+y*maxQ)
            sess.run(train,feed_dict={X:[state1],act:[np.identity(num_moves)[a]],Y:[y]})
            state = state1
            if done:
                break
            if j%100 == 99:
                saver.save(sess,model_fn)

