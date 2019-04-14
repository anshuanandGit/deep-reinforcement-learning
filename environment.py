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
maze_file_path= sys.argv[1]
output_file_path= sys.argv[2]
action_file_path= sys.argv[3]

class Environment:
    initial_x=0
    initial_y=0
    current_x= 0
    current_y= 0
    terminal_x= 0
    terminal_y= 0
    maze_row_cnt=0
    maze_col_cnt=0
    no_state=0
    no_action=0
    states=[]
    action_space=[0,1,2,3] #0 - west/left , 1- north/up , 2- east/right , 3- south/down
    no_action=len(action_space)
    maze=[]
    #parameters
    imdt_reward=-1
    trans_prob = 1
    
    #Method
    #This is Constructor class
    def __init__(self, filename):     
        self.maze =[]
        self.initial_x=0
        self.initial_y=0
        self.current_x= 0
        self.current_y= 0
        self.terminal_x= 0
        self.terminal_y= 0
        self.maze_row_cnt=0
        self.maze_col_cnt=0
        self.no_state=0        
        self.states=[]
    
        #load maze
        with open(filename, 'r') as fileObj:
            for line in fileObj:
                line=line.strip()        
                row =[]                
                for ch in line: 
                    self.states.append(ch)
                    row.append(ch)
                self.maze.append(row)
                self.maze_row_cnt = self.maze_row_cnt+1 #load row in maze
                

       #get no of states
        self.no_state=len(self.states)
        self.maze_col_cnt = int(self.no_state)/int(self.maze_row_cnt)        
        self.initialize_maze()#initialize maze
    
    #Method
    #This method intilalizes the maze   
    def initialize_maze(self):  
        r=int(self.maze_row_cnt)
        c=int(self.maze_col_cnt)
        for i in range(r):
            for j in range(c):
                if(self.maze[i][j] == 'S'):
                    self.current_x = i
                    self.current_y = j
                    self.initial_x= self.current_x
                    self.initial_y= self.current_y
                elif (self.maze[i][j] == 'G'):
                    self.terminal_x = i
                    self.terminal_y = j
                    
    
    #Method
    #This is simulates the action step on maze 
    def step(self,action):
        row = self.current_x
        col = self.current_y
        new_col = -1
        new_row = -1
        isTerminal =0
        
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
            if col ==self.maze_col_cnt -1:
                new_col=col
                new_row=row
            else:
                new_col=col + 1
                new_row=row
        #down/south
        elif action == 3:        
            if row ==self.maze_row_cnt-1:
                new_col=col
                new_row=row
            else:
                new_col=col
                new_row=row +1
    
        #special case to handle blockage on maze
        if self.maze[new_row][new_col] =='*':
            new_row = row
            new_col = col
        elif self.maze[new_row][new_col] =='G':
            isTerminal = 1
		
		#update the maze with latest position
        self.current_x = new_row
        self.current_y = new_col
        
        result= str(new_row) +" "+ str(new_col) +" "+ str(self.imdt_reward) +" "+ str(isTerminal)
        return new_row,new_col,self.imdt_reward,isTerminal
    
    #Method
    #This is simulates the action step on maze 
    def reset(self):
        self.current_x= self.initial_x
        self.current_y= self.initial_y
        return self.current_x,self.current_y
    

    
#--------------------------------------------------------------------------
#test the class created above
#openfiles....

if __name__ == "__main__":
    
    output_file = open(output_file_path,"w")

    action_from_file=[]
    with open(action_file_path, 'r') as fileObj:
        for line in fileObj:
            line=line.strip()
            acts= line.split()
            for act in acts:
                action_from_file.append(int(act))

    #initialize the Environment class 
    env = Environment(maze_file_path)

    for act in action_from_file:
        x,y,r,t = env.step(act)
        result= str(x) +" "+ str(y) +" "+ str(r) +" "+ str(t)
        output_file.writelines(result +'\n')

    #close all files
    output_file.close()


        