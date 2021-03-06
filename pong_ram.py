import gym
import tensorflow as tf
import numpy as np
import cv2
import random
import os
import pylab as pl

height = 128 # 21
width = 1 # 16
num_moves = 6
num_obs = 4
move_strength = [0.25, 0.5, 0.75, 1.0]
learning_rate = 0.01
# CartPole-v0, Berzerk-v0, SpaceInvadersDeterministic-v0
game_name = 'Pong-ram-v0'
model_fn = '/checkpoint/'+game_name+'/'+game_name+'.ckpt'

def cnn(X,act):
    #X = tf.reshape(X,shape=[-1, num_obs, height, width, 1])
    #c1 = tf.layers.conv3d(X, 16, (3,5,5),activation=tf.nn.relu)
    #c2 = tf.layers.conv3d(c1, 32, (1,3,3), activation=tf.nn.relu)
    #X = tf.reshape(X,shape=[-1, -1, height, 1])
    fc = tf.contrib.layers.flatten(X)
    fc = tf.concat([fc,act],1)  #
    fc = tf.layers.dense(fc,50,activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc,10,activation=tf.nn.relu) # ,activation=tf.nn.softmax
    fc3 = tf.layers.dense(fc2,1)
    return fc3

X = tf.placeholder(tf.float32, [None, num_obs, height])
act = tf.placeholder(tf.float32, [None, num_moves+1])
Y = tf.placeholder(tf.float32)  # (1-a)*Q(s,act)+a*(r+y*maxQ(s',act'))
out_ = cnn(X,act)
#predict = tf.argmax(out_,1)

loss = tf.losses.mean_squared_error(out_,Y)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

def get_state(env,prev,a,rep):
    rep = int(rep)
    res = [p for p in prev]
    last = [np.array(env.step(a))[:-1] for i in range(rep)]
    lr = len(res)
    ll = len(last)
    shf = lr-ll
    for i in range(shf,lr):
        res[i] = last[i-(shf)]
    for r in res:
        r[0] = r[0]/255.0
    return res

def ousoria(d):
    done = False
    for a in d:
        done = done or a
    return done

def norm_p(v):
    #m = np.min(v)
    #norm = [a-m+0.0001 for a in v]
    #return norm/np.sum(norm)
    norm = np.exp(v)
    return norm/np.sum(norm)

def get_act():
    l = len(move_strength)
    act_ = np.zeros((num_moves*l,num_moves+1))
    for i in range(num_moves):
        for j in range(l):
            act_[l*i+j][i] = 1.0
            act_[l*i+j][num_moves] = move_strength[j]
    return act_

def unpack_st(st):
    state = [s[0] for s in st]
    r = [s[1] for s in st]
    d = [s[2] for s in st]
    return state,r,d

env = gym.make(game_name)
y = 0.99
e = 0.1
alpha = 1.0
with tf.Session() as sess:
    saver = tf.train.Saver()
    if os.path.isfile(model_fn+'.meta'):
        saver.restore(sess,model_fn)
    else:
        sess.run(tf.global_variables_initializer())
    act_ = get_act()
    lms = len(move_strength)
    for t in range(5000):
        done = False
        obs = env.reset()
        prev = [np.array(env.step(0))[:-1] for i in range(num_obs)]
        for p in prev:
            p[0] = p[0]/255.0
        st = get_state(env,prev,0,1)
        state,r,d = unpack_st(st)
        for j in range(200000):
            env.render()
            allQ = sess.run([out_],feed_dict={X:[state for i in range(num_moves*lms)],act:act_}) #
            allQ = np.transpose(allQ[0])[0]
            #a = np.argmax(allQ)/lms
            #rep = np.argmax(allQ)%lms
            #a = np.random.choice(range(lms*num_moves),1,p=norm_p(allQ))[0]
            mov = np.random.choice(np.flatnonzero(allQ == allQ.max()))
            a = mov/lms
            rep = mov%lms
            if np.random.rand(1) < e: #  or a == 0 np.random.rand(1) < e
                a = env.action_space.sample()
                rep = np.random.choice(range(lms))
                #a = random.randint(2,5)
            new_st = get_state(env,prev,a,lms*move_strength[rep])  # 
            state1,r,d = unpack_st(new_st)
            total_r = np.sum(r)
            done = ousoria(d)
            print(a,rep,allQ)
            #print(a,total_r,allQ[a])
            maxQ = sess.run([out_],feed_dict={X:[state1 for i in range(num_moves*lms)],act:act_})
            maxQ = np.transpose(maxQ[0])[0]
            maxQ = np.max(maxQ)
            #maxA = np.random.choice(range(num_moves),1,p=norm_p(maxQ))[0]
            y = (1.0-alpha)*allQ[a] + alpha*(total_r+y*maxQ)
            if done:
                y = total_r
            sess.run(train,feed_dict={X:[state1],act:[act_[a*lms+rep]],Y:[y]})
            state = state1
            prev = new_st
            if done:
                break
            if j%100 == 99:
                saver.save(sess,model_fn)

