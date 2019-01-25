import gc
import importlib
import random
import numpy as np
import tensorflow as tf
from collections import deque
import itertools
import yrd_main
importlib.reload(yrd_main)
from yrd_main import *  # This is the environment of the shunting yard

from map_flat import *
from global_const import *
from action_decoder import label_decoder
import datetime
#import pandas as pd
#pd.set_option('mode.chained_assignment', 'raise')
import time
#import operator
from copy import deepcopy

#import pyodbc
import heapq
#from tensorflow.python.client import timeline


print('done importing')
#connect dataset

dataset_dic={}
for i in [0,1]:   
    with open("Data_object/states_"+str(i)+".pickle",'rb') as file:
        dataset=pickle.load(file)
    dataset_dic[i]=dataset
edame=1

while edame:
        done,solved,score=0,0,0
        #class_num_train=input("class_num_train?")
        class_num_train=1
        data=dataset_dic[int(class_num_train)]
        length_data=len(data)
        idx=random.randrange(length_data)
        yrd=pickle.loads(pickle.dumps(data[idx], -1))
        yrd.train_spec_map='in_place'
        mapinput=object
        self_map=Map(mapinput,None)


        while not done:
                
                state=self_map.rep(yrd)
                print(state.shape())
                print(state)
                #get action
                input_action_type=input("Move or Wait [M/W]")
                if input_action_type=='W':
                        action=Wait()
                else:
                        input_start_track=int(input('start track?'))
                        input_end_track=int(input('end track?'))
                        action=Move(None,None,None,input_start_track,input_end_track)
                    

                #change env
                print('getting next state and reward')
                solved, done, reward,violation_code,violation_string,train_id, train_material=yrd.next_state(action)
                score+=reward
                #print result
                print('solved:',solved,'done:',done,'reward of step',reward,'score',score,
                      'violation_code',violation_code,'violation_string',violation_string)
        edame=input("Continue?")

