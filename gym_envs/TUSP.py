# This file is part of Bootstrapped Dual Policy Iteration
# 
# Copyright 2018, Vrije Universiteit Brussel (http://vub.ac.be)
#     authored by Hélène Plisnier <helene.plisnier@vub.be>
#
# BDPI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BDPI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BDPI.  If not, see <http://www.gnu.org/licenses/>.

import gym
from gym import spaces
import math
import random
import sys
import numpy as np
import pandas as pd
import operator
from copy import deepcopy
import pickle

# This maps the environment to state (1D)
from gym_envs.map_flat import *
#This is requried to read data files(python object holding starting conditions of the night)
from yrd_main import *
#Infrastructure data
from gym_envs.global_const import *
#Get a summary of environment and return possible action indexs
from  gym_envs.action_decoder_reverse import *
#Translate action index to action input by the environment
from gym_envs.action_decoder import label_decoder


class TUSP(gym.Env):
    def __init__(self, rnd):
        #read data
        self.dataset_dic={}
        for i in [3]:   #where i is class of problmes by number of trains: 0 for [1,2,3], 1 for [4,5,6,7], 2 for [8,9,10,11,12], 3 for [13,14,15], 4 for [16,17,18]]
            with open("gym_envs/data_object/states_"+str(i)+".pickle",'rb') as file:
                dataset=pickle.load(file)
            self.dataset_dic[i]=dataset

            
        # 53 actions: track to track movements plus wait
        #self.action_size = [16]*[2 3 4 5 6 7 8  9]+[2 3 4 5 6 7 8  9]*[16]+[6 7 8  9]*[11 12]+[11 12]*[6 7 8 9]+
        #+[13]*[2 3 4 5 6 7 8  9]+[2 3 4 5 6 7 8  9]*[13]+[11 12]*[13]+[13]*[11 12]+wait
        self.action_space = spaces.Discrete(53)  
        self.observation_space = np.zeros([1610,])
        self.rnd = rnd
        self._timestep = 0
        self.reset()
        self.state_order = 0
        self.state_num = None
        self.train_idx = None
        self.tracks = {}            #holds position of the trains on the tracks
        self.not_arrived = []       #list of trains not yet arrived
        self.triggers = []          #list of triggers
        self.services = []          #panda dataframe for required time per train per service 
        self.action = None          #object specifing actions by agent
        self.action_sec = None      #changes derived from a taken action by agent(i.e. train being in service after moved to a track)

        self.track_length = []      #dictiononary holding track length
        self.material_length=[]     #dictiononary holding train type length



    #puts train on gate track when an arrival is due
    def _arrival(self,c):
        arrival = self.triggers[c]
        # find the train in 'not_arrived' list and put it on the arrival track
        for i in range(len(self.not_arrived)):
            if self.not_arrived[i].train == arrival.train:
                train = self.not_arrived.pop(i)
                # put this train on arrival track
                self.tracks[GATE_TRACK] = [train]
                return None
        assert False, 'Arrive error: train not found'
        return None

    def _departure(self):
        #in departure trigger, do nothing because all the things will be done in depart action
        return None

    #remove EndService trigger from trigger list
    def _end_service(self, trigger_num):
        end_service = self.triggers.pop(trigger_num)
        reward=0
        # find the train and turn off train.in_service
        for track in self.tracks:
            for train in self.tracks[track]:
                if train.train == end_service.train:
                    assert train.in_service != 0, 'EndService error: not in service'
                    reward_service= reward_fun_services(
                                self.services.loc[end_service.train, end_service.service])
                    reward+=reward_service
                    train.in_service = 0
                    self.services.loc[end_service.train, end_service.service] = 0
                    return reward
        assert False, 'EndService error: train not found'
        return None

    #for current action architecture is not required.
    #if actions are train/track or any setting where you should check whether the movement is allowed
    #,meaning origin destination tracks are connected and train is not blocked by other trains,
    #this function should be used
    #uncomment parts of code
    def _track_reachable(self, direction, block_li):
##        #block_li contains tracks that having trains on it and cannot get through(but can still be destination)
##        block_mat = np.diag(
##            [0 if i in block_li else 1 for i in range(np.shape(AB_MAT)[0])])
        mat = AB_MAT.copy()
        if direction == 'BA':
            mat = mat.T
##         hopmat_li = [mat.copy()]
##         # reachability is checked by: iteratively performs AB_mat*block_mat*AB_mat until it does not expand
##         while True:
##             if ((sum(hopmat_li) > 0) == ((sum(hopmat_li)+self._next_hop(hopmat_li[-1], mat, block_mat)) > 0)).all():
##             return (sum(hopmat_li) > 0).astype(int)
##             else:
##             hopmat_li.append(self._next_hop(hopmat_li[-1], mat, block_mat))
        return mat

    def _next_hop(self, hopmat, mat, block_mat):
        return np.dot(np.dot(hopmat, block_mat), mat)


    #function doing movement action
    def _move(self):
        move = self.action
        moved = 0
        fail_in_move, reward = 0, 0
        temp=None
        in_service, service_req=None,None
        move_time=0
        violation_code,violation_string=None,None
        train_id=None
        material_id=None

        #check for feasability of the movement (based on yard state)
        if move.start_track not in self.tracks:
            fail_in_move = 1
            violation_code,violation_string=1002,'track empty'
            return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id  ,in_service, service_req
        for i in NO_PARKING_TRACKS:
            if i in self.tracks:
                if move.start_track != i:
                    fail_in_move = 1
                    #print("Not allowed to park in Track" + str(i))
                    violation_code,violation_string=1003,'parked_in_13_16'
                    return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 
                break

        # when the train is found, figure out it is a left/right move
        left = self._track_reachable('BA', list(self.tracks))[
            move.start_track, move.end_track] == 1
        right = self._track_reachable('AB', list(self.tracks))[
            move.start_track, move.end_track] == 1
        # when executing the move, pop this train on the current track,
        # then append it on the correct position of the destination track
        if left:
            i = 0
            moved = 1
            temp = self.tracks[move.start_track].pop(i)

            if len(self.tracks[move.start_track]) == 0:  # delete key if empty
                self.tracks.pop(move.start_track)

            if move.end_track not in self.tracks:  # create key if does not exist
                self.tracks[move.end_track] = []
            self.tracks[move.end_track].append(temp)
        elif right:
            i = len(self.tracks[move.start_track]) - 1
            moved = 1
            temp = self.tracks[move.start_track].pop(i)
            if len(self.tracks[move.start_track]) == 0:
                self.tracks.pop(move.start_track)
            if move.end_track not in self.tracks:
                self.tracks[move.end_track] = []
            self.tracks[move.end_track].insert(0, temp)
        else:
            fail_in_move = 1
            print('AB matrix may at fault, access to track denied')
            return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 
        train_id=temp.train
        material_id=temp.material


        #check for feasability of the movement (based on train)
        for service in list(self.services.columns.values):
            service_req= self.services.loc[temp.train, service]
            in_service=((service_req>0) and (move.start_track in [11,12]))
        if temp.in_service != 0:
            fail_in_move = 1
            violation_code,violation_string=1004,'moved_while_in_service'
            return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 
        track_length_used = 0
        for train in self.tracks[move.end_track]:
            track_length_used += train.length
        if self.track_length[move.end_track] < track_length_used:
            'Move error: track length exceeded'
            if self.track_length[move.end_track] == 0:
                if len(self.tracks[move.end_track]) > 1:
                    fail_in_move = 1
                    violation_code,violation_string=1005, 'track_max_num_train_exceeded'
                    return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 
            else:
                fail_in_move = 1
                violation_code,violation_string=1006, 'track_length_exceeded'
                return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 
        if moved == 1:
            # Depart if the end_track in GATE_TRACK
            if move.end_track == GATE_TRACK:  
                self.action_sec = Depart(temp.train,None)
                fail_in_move,violation_code,violation_string,partial_reward = self._depart()
                if fail_in_move == 0:
                    reward = 2+partial_reward
                else:
                    reward=partial_reward
                    return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req
            # give negative reward end_track in relocation track
            elif move.end_track == 13:
                reward += -0.3
            #if arrival is handled on time remove Arrival trigger from trigger list
            if move.start_track==GATE_TRACK:
                    assert isinstance(self.triggers[0], Arrival), 'arrival check not working'
                    self.triggers.pop(0)
                    reward=.2
            #if destination track is cleaning track and train requires cleaning, strat cleaning
            for service in list(self.services.columns.values):
                req_time = self.services.loc[temp.train, service]
                # start_service if on right track
                if (move.end_track in SERVICE_TRACK[service]) and (req_time > 0):
                    self.action_sec = StartService(temp.train, service, req_time, None)
                    self._start_service()
                #if destination track is cleaning track and train does not require cleaning, give negative reward
                if (move.end_track in SERVICE_TRACK[service]) and (req_time == 0):
                    reward+=-0.5
        if fail_in_move==0:
            move_time=3 
        return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 

    def _depart(self):
        reward=0
        violation_code,violation_string=None,None
        depart_action=self.action_sec
        fail_in_move = 0
        #check time for departure
        if (isinstance(self.triggers[0], Departure) != True) | (self.triggers[0].time != 0):
            fail_in_move = 1
            violation_code,violation_string=1001,'not_departure_time'
            return fail_in_move,violation_code,violation_string,reward
        #reward+=.5 #while you do not filter on your actions it is better to give  areward here
        depart = self.action_sec
        track=GATE_TRACK
        i=0
        #check type for departure
        if (self.tracks[track][i].material != self.triggers[0].material):
            fail_in_move = 1
            violation_code,violation_string=1007,'wrong_material'
            return fail_in_move,violation_code,violation_string,reward
        reward+=0.5
        #check type for departure
        if (self.tracks[track][i].in_service != 0) or ((max(self.services.loc[depart.train]) != 0)):
            fail_in_move = 1
            #print('servise not done')
            violation_code,violation_string=1008,'depart_with_undone_service'
            return fail_in_move,violation_code,violation_string,reward

        #when you arrive here departure has beeen done correctly
        self.tracks[track].pop(i)
        if len(self.tracks[track]) == 0:  # delete the list
            self.tracks.pop(track)
        assert isinstance(self.triggers[0], Departure), 'departure check not working'
        self.triggers.pop(0)
        return fail_in_move,violation_code,violation_string,reward


    def _start_service(self):
        fail_in_move=0
        start_service = self.action_sec
        task_time=start_service.task_time
        # find the train
        for track in self.tracks:
            for train in self.tracks[track]:
                if train.train == start_service.train:
                    self.services.loc[start_service.train, start_service.service]+=3
                    # turn the 'in_service' sign on the train to True
                    # add the corresponding EndService into triggers
                    assert train.in_service == 0, 'StartService error: in service'
                    assert track in SERVICE_TRACK[start_service.service], 'StartService error: wrong service-track'
                    assert self.services.loc[start_service.train,
                                             start_service.service] != 0, 'StartService error: service not required'
                    # set train to in_service, add trigger, delete service
                    #increase time of end service trigger by 3, since 3 minutes will pass after this to do the movement
                    assert task_time+3==self.services.loc[start_service.train, start_service.service],'task times do not match'
                    train.in_service = start_service.service
                    self.triggers.insert(0, EndService(
                        train.train, self.services.loc[start_service.train, start_service.service], start_service.service))
                    self.triggers.sort(key=operator.attrgetter('time'))
        return fail_in_move

    def _wait(self):
        duration=0
        reward=0
        violation_code,violation_string=None,None
        fail_in_wait=0
        #check weather wait movement is allowed in this state.
        #when actions are filterd there is no need.
        for i in NO_PARKING_TRACKS:
            if i in self.tracks:
                    violation_code,violation_string=1013,'wait_while_train_in_13_16'
                    fail_in_wait=1
                    return duration, fail_in_wait, reward,violation_code,violation_string
        duration = max(0,self.triggers[0].time)
        if duration==0:
            fail_in_wait=1
            violation_code,violation_string=1011,'wait_duration_is_0'
            #print('wait_duration_is_0')
            return duration, fail_in_wait, reward,violation_code,violation_string


        #find wait duration and return it
        if duration> 60:
                duration=60
                return duration, fail_in_wait, reward,violation_code,violation_string
        else:
                c=min(2,len(self.triggers)-1)
                while c>-1:
                        if isinstance(self.triggers[c], Arrival) and self.triggers[c].time==duration:
                            self._arrival(c)
                        elif isinstance(self.triggers[c], Departure) and self.triggers[c].time==duration:
                            self._departure()
                        elif isinstance(self.triggers[c], EndService)and self.triggers[c].time==duration:        
                            reward=self._end_service(c)
                            c+=-1
                        c+=-1
                return duration, fail_in_wait, reward,violation_code,violation_string


    #function used in step. takes action returns new state of yard    
    def next_state(self, action=None):
        violation_code,violation_string=None,None
        in_service, service_req=None,None
        self.triggers.sort(key=operator.attrgetter('time'))
        done = 0
        solved = 0
        reward = 0
        fail_in_move=0
        fail_in_wait=0
        train_id=None
        material_id=None
        if action != None:
            self.action = action
        else:
            assert self.action != None, 'action none'
##        print(self.action)
##        print(id(Wait), id(action.__class__))
##        print(id(Move), id(action.__class__))
        if isinstance(self.action, Wait):
            duration, fail_in_wait, reward,violation_code,violation_string  = self._wait()
        elif isinstance(self.action, Move):
            duration, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req = self._move()
        else:
            assert False, 'unknown action type'

        
        fail_in_action=max(fail_in_wait,fail_in_move)
        if fail_in_action:
                done=1
        #if no violations so far, froward yard in time, and update requried services 
        else:
                counter=len(self.triggers)
                i=0
                while i <counter:  # update service time required
                    trigger = self.triggers[i]
                    if isinstance(trigger, EndService):
                        reward_service=0
                        if self.services.loc[trigger.train, trigger.service] > duration:
                            self.services.loc[trigger.train,
                                               trigger.service] -= duration
                            reward_service= reward_fun_services(duration)
                            reward+=reward_service
                        elif self.services.loc[trigger.train, trigger.service] > 0:
                            reward_service= reward_fun_services(
                                self.services.loc[trigger.train, trigger.service])
                            reward+=reward_service

                            self.services.loc[trigger.train, trigger.service] = 0
                            self._end_service(i)
                            counter-=1
                            i-=1
                    i+=1

                # pass time
                #(If endservicee happened while moving ignore)
                counter =len(self.triggers)
                i=0
                while i<counter:
                    if isinstance(self.triggers[i], EndService):
                        if self.triggers[i].time < duration:
                                self.triggers.pop(i)
                                counter-=-1
                                i-=-1
                        else:
                                self.triggers[i].time -= duration
                    else:
                        self.triggers[i].time -= duration
                        if isinstance(self.triggers[i], Arrival)==1 and self.triggers[i].time==0 and isinstance(self.action, Move):
                                self._arrival(i)
                    i+=1

                #check for solved or a missed arrival and departure
                solved, done,violation_code_eval,violation_string_eval = self._evaluate()
                if solved:
                        reward+=5
                if violation_code==None:
                        violation_code=violation_code_eval
                        violation_string=violation_string_eval
        
        self.state_order += 1
        # clear the action
        self.action = None
        pre_done=0
        if solved ==0:
                if done==0:
                        if isinstance(self.triggers[0], Departure)==1 and self.triggers[0].time==0:
                                pre_done=1
                                for track in self.tracks:
                                        if track in list(range(2,10)):
                                                train=self.tracks[track][0]
                                                if train.material == self.triggers[0].material:
                                                        pre_done=0
        if pre_done==1:
                done=1
                violation_code,violation_string=1012,'no_train_can_depart'

        self.triggers.sort(key=operator.attrgetter('time'))  
        return solved, done, reward,violation_code,violation_string,train_id,material_id, in_service, service_req



    #gives a summary of the yard
    #used to filter actions
    #returns a list of tracks with atleast one train
    #in addtion to information indicating if it is time for departure or arrival
    def _get_not_empty_tracks(self,report_error):
        not_empty_tracks_and_wait_possible=[]
        
        for i in self.tracks:
                not_empty_tracks_and_wait_possible+=[i]
        if  (self.triggers[0].time>0) or (isinstance(self.triggers[0], Departure)==0 and isinstance(self.triggers[0], Arrival)==0):
                not_empty_tracks_and_wait_possible+=[100]
        if (isinstance(self.triggers[0], Departure)==1 and self.triggers[0].time==0):
                not_empty_tracks_and_wait_possible+=['d']
        right=0


        #to check for possible errors in dynamics 
        if report_error==1:
                if 'd' in not_empty_tracks_and_wait_possible:
                    right+=1
                if 16 in not_empty_tracks_and_wait_possible:
                    right+=1
                if 100 in not_empty_tracks_and_wait_possible:
                    right+=1
                if right!=1:
                        print('dynamics of yard has gone wrong',not_empty_tracks_and_wait_possible)
                        print('Arrival',isinstance(self.triggers[0], Arrival))
                        print('Departure',isinstance(self.triggers[0], Departure))
                        print(self.triggers[0].time)
        return not_empty_tracks_and_wait_possible


    #gives a summary of the yard
    #not used in training, but to analyze made planes
    #returns a number of left departures, number of trains in the yard, number of trains in a specific track,
    #information on next trigger, and time (minutes passed) from the start of the night (8:00 am)
    def _get_state_summary(self,start_track):
        num_dep=0
        for trigger in self.triggers:
                if isinstance(trigger, Departure):
                        num_dep+=1                           
        num_train_yrd=0
        for track in self.tracks:
                for train in self.tracks[track]:
                        num_train_yrd+=1
        num_train_s_track=0
        next_trigger=None
        if start_track>0:
                for train in self.tracks[start_track]:
                        num_train_s_track+=1
        if isinstance(self.triggers[0], Arrival):
                next_trigger='Arrival'
        elif isinstance(self.triggers[0], Departure):
                next_trigger='Departure'
        else:
                next_trigger='End_service'
        time_to_trigger=self.triggers[0].time
        time=self.time 
        return num_dep, num_train_yrd, num_train_s_track, next_trigger, time_to_trigger, time


    #check if solved, and check time for arrival and departure have not passed
    def _evaluate(self):  
        violation_code,violation_string=None,None
        if self.triggers == []:
            return 1, 1,violation_code,violation_string  # solved
        else:
            for t in self.triggers:
                if t.time < 0:
                    if isinstance(self.triggers[0], Arrival):
                        violation_code,violation_string=1009,'missed_arrival'
                    if isinstance(self.triggers[0], Departure):
                        violation_code,violation_string=1010,'missed_departure'
                    return 0, 1,violation_code,violation_string  # failed
            return 0, 0,violation_code,violation_string  # not terminal

    def reset(self):
        """ Reset the environment and return the initial state number"""
        #class_num_train=random.randrange(2)
        class_num_train=3
        data=self.dataset_dic[class_num_train]
        length_data=len(data)
        # pick a randome episode from the class of problem you want
        length_data=1000 #!!!to over fit to a small set
        idx=random.randrange(length_data)
        yrd=pickle.loads(pickle.dumps(data[idx], -1))
        self.state_order = yrd.state_order
        self.state_num = yrd.state_num
        self.train_idx = yrd.train_idx
        self.tracks = {}
        self.not_arrived = yrd.not_arrived
        self.triggers = yrd.triggers
        self.services = yrd.services
        self.action = None
        self.action_sec = None
        self.track_length = yrd.track_length
        self.material_length=yrd.material_length
        mapinput=object
        self_map=Map(mapinput,None)
        state=self_map.rep(self)
        return state

    def step(self, action_index):
        action=label_decoder(action_index)
        solved, terminal, reward, violation_code,violation_string,train_id, train_material, in_service, service_req =self.next_state(action)
        if solved==0 :
            test_possible_actions=label_decoder_reverse(self._get_not_empty_tracks(0))
            if len(test_possible_actions)==0 and terminal==0:
                terminal=1
        if terminal==0:
            mapinput=object
            self_map=Map(mapinput,None)
            state=self_map.rep(self)
        else:
            state=np.zeros([1610,])
            
        # Return the current state, a reward and whether the episode terminates
        return state, reward, terminal, {}




















