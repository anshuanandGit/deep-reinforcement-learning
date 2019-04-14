
# coding: utf-8

# In[85]:


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

#commandline param
maze_file= sys.argv[1]
value_output_file= sys.argv[2]
q_value_output_file= sys.argv[3]
policy_output_file= sys.argv[4]
num_epoch= sys.argv[5]
disc_fctr= sys.argv[6]

start = time.time()

maze_row_cnt=0
maze_col_cnt=0
no_state=0
no_action=0
states=[]
action=[0,1,2,3] #0 - west/left , 1- north/up , 2- east/right , 3- south/down
no_action=len(action)
maze=[]
q_matrix=[]

#parameters
imdt_reward=-1
trans_prob = 1

#given state and action, return reward and next state
def get_s_prime(row,col, action):
    #row = int(current_state)/int(maze_col_cnt)
    #col = int(current_state) % int(maze_col_cnt)
    new_col = -1
    new_row = -1
    #left/west
    if action==0:           
        if col ==0:
            new_col=col
            new_row=row
        else:
            new_col=col -1
            new_row=row
     #up/north        
    elif action ==1:         
        if row ==0:
            new_col=col
            new_row=row
        else:
            new_col=col
            new_row=row -1
    #right/east
    elif action == 2:         
        if col ==maze_col_cnt -1:
            new_col=col
            new_row=row
        else:
            new_col=col + 1
            new_row=row
    #down/south
    elif action == 3:        
        if row ==maze_row_cnt-1:
            new_col=col
            new_row=row
        else:
            new_col=col
            new_row=row +1
    
    #special case to handle blockage on maze
    if maze[new_row][new_col] =='*':
        new_row = row
        new_col = col
        
    return new_row, new_col

#calcualte value iteration
def apply_value_iteration():
    rwcnt = int(maze_row_cnt)
    clcnt = int(maze_col_cnt)
    vs = np.zeros(shape=(rwcnt,clcnt))
    pi = np.zeros(shape=(rwcnt,clcnt))
    qsa =[]
    
    for i in range(int(num_epoch)):
        qsa_mat=[]
        vp = np.copy(vs)
        for m in range(rwcnt):
            q_at_row=[]
            for n in range(clcnt):
                q_at_col=np.zeros(no_action)
                for a in range(no_action):
                    x_prime, y_prime = get_s_prime(m,n,a) # get prime state for given action
                    if(maze[m][n] =='G'):
                        q_at_col[a] =0
                    elif(maze[m][n] !='*'):
                        q_at_col[a] = float(imdt_reward )+ float(disc_fctr) *float(trans_prob) * float(vp[x_prime][y_prime])
                #qsa[m][n] =q
                vs[m][n] =max(q_at_col)
                pi[m][n] = np.argmax(q_at_col)
                q_at_row.append(q_at_col)
                
            qsa_mat.append(q_at_row)
    qsa = qsa_mat
    return vs , pi, qsa               
                    
#write output to file    v_file.writelines(format(prob, '.20f') + " "+'\n')
def writefile(v,p,q):
    rwcnt = int(maze_row_cnt)
    clcnt = int(maze_col_cnt)
    
    for i in range(rwcnt):
        for j in range(clcnt):
            if(maze[i][j] !='*'):
                v_file.writelines(str(i) + " " + str(j) +" " + str(v[i][j]) +'\n')
                p_file.writelines(str(i) + " " + str(j) +" " + str(p[i][j]) +'\n')
            
                a =q[i][j]
                for k in range(no_action):
                    q_file.writelines(str(i) + " " + str(j) +" " + str(k) +" "+ str(a[k]) +'\n')

#--------------------------------------------------------------------------------------------------------                    
#load all words
#openfiles....
v_file = open(value_output_file,"w")
q_file = open(q_value_output_file,"w")
p_file = open(policy_output_file,"w")    
    
#load maze
with open(maze_file, 'r') as fileObj:
    for line in fileObj:
        line=line.strip()        
        row =[]
        for ch in line:            
            states.append(ch)
            row.append(ch)
        #load row in maze
        maze_row_cnt=maze_row_cnt+1
        maze.append(row)

#get no of states
no_state=len(states)
maze_col_cnt = int(no_state)/int(maze_row_cnt)

#print(maze_row_cnt) 
v,p,q = apply_value_iteration()
writefile(v,p,q)
        
        

#close all files
v_file.close()
q_file.close()
p_file.close()
# run your code
end = time.time()

elapsed = end - start
print("done in : %f" %(elapsed))

