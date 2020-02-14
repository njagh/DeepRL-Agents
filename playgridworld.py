import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from Qnetwork import Qnetwork
from Qnetwork import experience_buffer

from gridworld import gameEnv

env = gameEnv(partial=False, size=5)
path = "/mnt/chromeos/removable/SDCard/data/dqn/medium" #The path to save ou
print('going to reset default graph')
tf.reset_default_graph()
print('going to define mainQN')
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
mainQN = Qnetwork(h_size)
print('going to define targetQN')
targetQN = Qnetwork(h_size)
print('init global variables')
init = tf.global_variables_initializer()
print('going to make saver')
saver = tf.train.Saver()
print('going to set trainable_variables')
trainables = tf.trainable_variables()

print('getting ckpt')
ckpt = tf.train.get_checkpoint_state(path)
print('going to restore')
sess = tf.Session() 
saver.restore(sess,ckpt.model_checkpoint_path)
print('restored')

def processState(states):
    return np.reshape(states,[21168])


maxsteps = 20
value = 'y'
while value == 'y': 
    n = 0
    rall = 0.
    d = False
    s = env.reset()
    s = processState(s)
    while n <= maxsteps:
        n += 1
        print(rall)
        #    plt.pause(.01)
        a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
        s ,r,d = env.step(action = a) #np.random.randint(0,4))
        rall += r
        s = processState(s)
        env.state = env.renderEnv()
        plt.imshow(env.state,interpolation="nearest")
        plt.pause(.001)
        if d == True:
            break
    value = input('press y to run again')
        
#env.step(action = 0)
#env.state = env.renderEnv()
#plt.imshow(env.state,interpolation="nearest")
#plt.pause(.01)
#print(env.objects[0].x, env.objects[0].y)

#env.step(action = 2)
#env.state = env.renderEnv()
#plt.imshow(env.state,interpolation="nearest")
#plt.pause(.01)
#print(env.objects[0].x, env.objects[0].y)



#plt.imshow(env.state,interpolation="nearest")

#plt.show()

#x = [1,2,3,4,5]
#plt.plot(x,x)
#plt.show(block = False)
#plt.plot([i - 1 for i in x], x)
#plt.show()


