
# coding: utf-8

# In[40]:


# Import the required librray
import sys
import numpy as np
import re
import csv
import json
eps = np.finfo(float).eps
from numpy import log2 as log
import time
import decimal
import random
from environment import Environment

#commandline param
maze_file_path= sys.argv[1]
value_output_file= sys.argv[2]
q_value_output_file= sys.argv[3]
policy_output_file= sys.argv[4]
episode_num= sys.argv[5]
episode_len= sys.argv[6]
learn_rate= sys.argv[7]
discnt_rt= sys.argv[8]
epsilon= sys.argv[9]

start = time.time()
# Create Environment Variable..
env = Environment(maze_file_path)
rwcnt = int(env.maze_row_cnt)
clcnt = int(env.maze_col_cnt)
no_action = int(env.no_action)

alpha = learn_rate


actions_seq = env.action_space

def prepare_qmatrix():
    qsa =[]
    q_a = np.zeros(no_action)
    
    for t in range(rwcnt):
        q_c=[]
        for u in range(clcnt):
            q_c.append(q_a)
        qsa.append(q_c)
    return qsa

def getindex(x,y):
    indx = clcnt * x +y
    return indx
    

#calcualte value iteration
def apply_q_learning():
    # preapre q matrix
    #qsa = prepare_qmatrix()
    qt=np.zeros([env.no_state,env.no_action])  
   
    for i in range(int(episode_num)):
        x, y = env.reset()        
        
        for j in range(int(episode_len)):
            
            state = env.maze[x][y]
            action=0
            if np.random.uniform(0, 1) < float(epsilon):
                    action = random.randint(0,3)  # Check the action space                    
            else:
                    action = np.argmax(qt[getindex(x,y)])  # Check the learned values,pick best move          
                    
            if(state == 'G'):                
                break
            else: 
                # take the step in environment to collect reward
                n_x, n_y, reward, is_terminal = env.step(action)             
                old_qas = qt[getindex(x,y)][action]
                next_max_qas = np.max(qt[getindex(n_x,n_y)])
               # Update the new value
                new_qas =  float(old_qas) + float(alpha)*(float(reward) + float(discnt_rt) * float(next_max_qas) -float(old_qas))
                qt[getindex(x,y)][action] = new_qas            
                x = n_x
                y = n_y                

              
    return qt


#this method extracts policy and value from q_value
def extract_policy_and_value(qsa):
    
    vs = np.zeros(shape=(rwcnt,clcnt))
    pi = np.zeros(shape=(rwcnt,clcnt))
    
    for i in range(rwcnt):
        for j in range(clcnt):
            indx = getindex(i,j)
            vs[i][j] = max(qsa[indx])
            pi[i][j] = np.argmax(qsa[indx])
    
    return vs, pi

#write output to file    v_file.writelines(format(prob, '.20f') + " "+'\n')
def writefile(v,p,q):    
    print(q)
    print(v)
    print(p)
    for i in range(rwcnt):
        for j in range(clcnt):
            if(env.maze[i][j] !='*'):
                v_file.writelines(str(i) + " " + str(j) +" " + str(v[i][j]) +'\n')
                p_file.writelines(str(i) + " " + str(j) +" " + str(p[i][j]) +'\n')
            
                indx = getindex(i,j)
                for k in range(no_action):
                    q_file.writelines(str(i) + " " + str(j) +" " + str(k) +" "+ str(q[indx][k]) +'\n')

#---------------------------------------------------------------------------------------------------------
#open other files..
v_file = open(value_output_file,"w")
q_file = open(q_value_output_file,"w")
p_file = open(policy_output_file,"w") 

qsa = apply_q_learning()
vs, pi = extract_policy_and_value(qsa)

writefile(vs,pi,qsa)

#close all files
v_file.close()
q_file.close()
p_file.close()
# run your code
end = time.time()

elapsed = end - start
print("done in : %f" %(elapsed))
