import pandas as pd
import numpy as np
import operator
from copy import deepcopy
#from gym_envs import global_const
from gym_envs.global_const import *
import pickle


def reward_fun_services(time):
        time = time / 60
        return time

class Train(object):
    def __init__(self, train, material, length):
        self.train = train
        self.material = material
        self.length = length
        self.in_service = 0


class Action(object):
    def __init__(self):
        pass


class Action_sec(object):
    def __init__(self):
        pass


class Move(Action):
    def __init__(self, train, time, end_time, start_track, end_track):
        self.train = train
        self.time = time
        self.start_track = start_track
        self.end_track = end_track

    def print_action(self):
        print('Move(', self.train, ', ', self.end_track, ')', sep="")


class Depart(Action_sec):
    def __init__(self, train, time):
        self.train = train
        self.time = time

    def print_action(self):
        print('Depart(', self.train, ')', sep="")


class StartService(Action_sec):
    def __init__(self, train, service,task_time, time):
        self.train = train
        self.service = service
        self.task_time=task_time
        self.time = time

    def print_action(self):
        print('StartService(', self.train, ', ', self.service, ')', sep="")


class Wait(Action):
    def __init__(self):
        pass

    def print_action(self):
        print('Wait')


class Trigger(object):
    def __init__(self):
        pass


class Arrival(Trigger):
    def __init__(self, train, time):
        self.train = train
        self.time = time

    def print_trigger(self):
        print('Arrival: train:', self.train, 'time:', self.time)


class Departure(Trigger):
    def __init__(self, material, time):
        self.material = material
        self.time = time

    def print_trigger(self):
        print('Departure: material:', self.material, 'time', self.time)


class EndService(Trigger):
    def __init__(self, train, time, service):
        self.train = train
        self.time = time
        self.service = service

    def print_trigger(self):
        print('EndService: train:', self.train, 'time', self.time)


class State(object):
    def __init__(self, instance):
        self.instance_id = instance.instance_id
        self.state_order = 0
        self.state_num = None
        self.train_idx = None

        self.tracks = {}
        self.not_arrived = instance.train_li

        self.triggers = instance.triggers
        self.services = instance.services
        self.action = None
        self.action_sec = None

        self.track_length = instance.track_length
        self.material_length=instance.material_length

    def print_state(self):
        print('State', self.state_order)
        print('')
        print('Tracks')
        for track in self.tracks:
            print(track, end=': ')
            for train in self.tracks[track]:
                print(train.train, train.in_service, end=',')
        print('')
        print('')
        print('Triggers')
        for i in self.triggers:
            i.print_trigger()
        print('')
        print('Services')
        print(self.services)
        if self.action != None:
            self.action.print_action()

    # triggers
    def _arrival(self,c):
        arrival = self.triggers[c]
        # find the train in 'not_arrived' list and put it on the arrival track
        for i in range(len(self.not_arrived)):
            if self.not_arrived[i].train == arrival.train:
                train = self.not_arrived.pop(i)
                # put this train on arrival track
                #assert GATE_TRACK not in self.tracks, 'Arrive error: arrival track occupied'
                self.tracks[GATE_TRACK] = [train]
                return None
        assert False, 'Arrive error: train not found'
        return None

    def _departure(self):
        # in departure trigger, do nothing because all the things will be done in depart action
        return None

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

    # actions
    def _track_reachable(self, direction, block_li):
        # block_li contains tracks that having trains on it and cannot get through(but can still be destination)
        block_mat = np.diag(
            [0 if i in block_li else 1 for i in range(np.shape(AB_MAT)[0])])
        mat = AB_MAT.copy()
        if direction == 'BA':
            mat = mat.T
        # hopmat_li = [mat.copy()]
        # # reachability is checked by: iteratively performs AB_mat*block_mat*AB_mat until it does not expand
        # while True:
            # if ((sum(hopmat_li) > 0) == ((sum(hopmat_li)+self._next_hop(hopmat_li[-1], mat, block_mat)) > 0)).all():
            # return (sum(hopmat_li) > 0).astype(int)
            # else:
            # hopmat_li.append(self._next_hop(hopmat_li[-1], mat, block_mat))
        return mat

    def _next_hop(self, hopmat, mat, block_mat):
        return np.dot(np.dot(hopmat, block_mat), mat)

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
        if move.start_track not in self.tracks:
            fail_in_move = 1
            #print("Track empty")
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
        #assert self.tracks[track][i].in_service == 0, 'Move error: in service'
        left = self._track_reachable('BA', list(self.tracks))[
            move.start_track, move.end_track] == 1
        right = self._track_reachable('AB', list(self.tracks))[
            move.start_track, move.end_track] == 1
        # when executing the move, pop this train on the current track,
        # then append it on the correct position of the destination track
        # also consume 3 min for the state
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
            #assert False, 'Move error: blocked'
            # for trigger in self.triggers:
            # trigger.time -= MOVE_TIME
            # """!!! to be added move_time=find"""
        train_id=temp.train
        material_id=temp.material
        for service in list(self.services.columns.values):
            service_req= self.services.loc[temp.train, service]
            in_service=((service_req>0) and (move.start_track in [11,12]))
        if temp.in_service != 0:
            fail_in_move = 1
            #print("train in servise" )
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
                    #print('no more than one train allowed')
                    violation_code,violation_string=1005, 'track_max_num_train_exceeded'
                    return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 
            else:
                fail_in_move = 1
                #print('track length exceeded')
                violation_code,violation_string=1006, 'track_length_exceeded'
                return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 
        if moved == 1:
            
            if move.end_track == GATE_TRACK:  # Depart if the end_track in GATE_TRACK
                self.action_sec = Depart(temp.train,None)
                fail_in_move,violation_code,violation_string,partial_reward = self._depart()
                if fail_in_move == 0:
                    reward = 2+partial_reward
                else:
                    reward=partial_reward
                    return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 
            elif move.end_track == 13:
                reward += -0.3
            if move.start_track==GATE_TRACK:
                    #remove arrival trigger
                    #print('gate_track_oc',gate_track_oc)
                    assert isinstance(self.triggers[0], Arrival), 'arrival check not working'
                    self.triggers.pop(0)
                    #gate_track_oc+=-1
                    reward=.2
            for service in list(self.services.columns.values):
                req_time = self.services.loc[temp.train, service]
                # start_service if on right track
                if (move.end_track in SERVICE_TRACK[service]) and (req_time > 0):
                    self.action_sec = StartService(temp.train, service, req_time, None)
                    self._start_service()
                if (move.end_track in SERVICE_TRACK[service]) and (req_time == 0):
                    reward+=-0.5
        if fail_in_move==0:
            #print('move_action:','train:',temp.train,'from:',move.start_track,'to',move.end_track)
            move_time=3 #!!!! to be changed
                    
        return  move_time, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req 
        #assert False, 'Move error: train not found'

    def _depart(self):
        reward=0
        violation_code,violation_string=None,None
        depart_action=self.action_sec
        fail_in_move = 0
        #assert isinstance(self.triggers[0], Departure), 'Depart error: next trigger not departure'
        if (isinstance(self.triggers[0], Departure) != True) | (self.triggers[0].time != 0):
            fail_in_move = 1
            #print('not time for departure')
            violation_code,violation_string=1001,'not_departure_time'
            return fail_in_move,violation_code,violation_string,reward
        #reward+=.5
        #assert self.triggers[0].time == 0, 'Depart error: not the depart time'
        depart = self.action_sec
        track=GATE_TRACK
        # find the train
        # for track in self.tracks:
        #    for i in range(len(self.tracks[track])):
        #        if self.tracks[track][i].train == depart.train:
        # pop the train from the current track but not putting it back
        # also consume 3 min for the state
        #assert self.tracks[track][i].material == self.triggers[0].material, 'Depart error: wrong material'

        #assert self._track_reachable('BA', list(self.tracks))[track, GATE_TRACK] == 1, 'Depart error: no available path'
        #assert i == 0, 'Depart error: blocked on current track'

        i=0
        if (self.tracks[track][i].material != self.triggers[0].material):
            fail_in_move = 1
            #print('wrong material')
            violation_code,violation_string=1007,'wrong_material'
            return fail_in_move,violation_code,violation_string,reward
        reward+=0.5
        if (self.tracks[track][i].in_service != 0) or ((max(self.services.loc[depart.train]) != 0)):
            fail_in_move = 1
            #print('servise not done')
            violation_code,violation_string=1008,'depart_with_undone_service'
            return fail_in_move,violation_code,violation_string,reward
        #reward+=.2
        #assert self.tracks[track][i].in_service == 0, 'Depart error: in service'+str(self.tracks[track][i].train)
        #assert (self.services.loc[depart.train] == 0).all(), 'Depart error: service remain'
        self.tracks[track].pop(i)
        # for trigger in self.triggers:
        #    trigger.time -= MOVE_TIME
        """!!! to be added move_time=find"""
        if len(self.tracks[track]) == 0:  # delete the list
            self.tracks.pop(track)
        assert isinstance(self.triggers[0], Departure), 'departure check not working'
        self.triggers.pop(0)
        return fail_in_move,violation_code,violation_string,reward
        #assert False, 'Depart error: train not found'
        # return None

    def _start_service(self):
        fail_in_move=0
        start_service = self.action_sec
        task_time=start_service.task_time
        # find the train
        for track in self.tracks:
            for train in self.tracks[track]:
                if train.train == start_service.train:
                    self.services.loc[start_service.train, start_service.service]+=3
                    #print(str(train.train),' going to service ',str(start_service.service))
                    # turn the 'in_service' sign on the train to True
                    # add the corresponding EndService into triggers
                    assert train.in_service == 0, 'StartService error: in service'
                    assert track in SERVICE_TRACK[start_service.service], 'StartService error: wrong service-track'
                    assert self.services.loc[start_service.train,
                                             start_service.service] != 0, 'StartService error: service not required'
                    # set train to in_service, add trigger, delete service
                    assert task_time+3==self.services.loc[start_service.train, start_service.service],'task times do not match'
                    train.in_service = start_service.service
                    #print('time to endservise triger',self.services.loc[start_service.train, start_service.service],'=',task_time)
                    self.triggers.insert(0, EndService(
                        train.train, self.services.loc[start_service.train, start_service.service], start_service.service))
                    self.triggers.sort(key=operator.attrgetter('time'))
                    # self.services.loc[start_service.train, start_service.service] = 0 #H: !!!! should not do this
        #assert False, 'StartService error: train not found'
        return fail_in_move

    def _wait(self):
        duration=0
        reward=0
        violation_code,violation_string=None,None
        # first skip time to reach next trigger
        #print('EndService:',isinstance(self.triggers[0], EndService))
        #print('Arrival:',isinstance(self.triggers[0], Arrival))
        #print('Departure:',isinstance(self.triggers[0], Departure))
        #print('self.triggers[0].time',self.triggers[0].time)
        fail_in_wait=0
        for i in NO_PARKING_TRACKS:
            if i in self.tracks:
                    violation_code,violation_string=1013,'wait_while_train_in_13_16'
                    fail_in_wait=1
                    return duration, fail_in_wait,violation_code,violation_string
        duration = max(0,self.triggers[0].time)
        
        if duration==0:
            fail_in_wait=1
            violation_code,violation_string=1011,'wait_duration_is_0'
            #print('wait_duration_is_0')
            return duration, fail_in_wait,violation_code,violation_string
        #print('wait for ',str(duration))
        # if isinstance(self.triggers[0], Departure) and duration == 0:
        # assert False, 'Wait error: should depart'
        #for i in self.triggers:
        #    i.time -= duration
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
        #                else:
        #                    assert False, 'Wait error: unknown trigger'

                return duration, fail_in_wait, reward,violation_code,violation_string

    def next_state(self, action=None):
        #if GATE_TRACK in self.tracks:
            #print('this is for debug ','length of gate_track ',len(self.tracks[GATE_TRACK]))
        #else:
            #print('this is for debug ','length of gate_track ',0)
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

        
        if isinstance(self.action, Wait):
            duration, fail_in_wait, reward,violation_code,violation_string  = self._wait()

        elif isinstance(self.action, Move):
            duration, fail_in_move, reward,violation_code,violation_string,train_id,material_id ,in_service, service_req = self._move()
        else:
            assert False, 'unknown action type'
        
        #print(duration, fail_in_move, reward,violation_code,violation_string)
        fail_in_action=max(fail_in_wait,fail_in_move)
        if fail_in_action:
                done=1
                #return solved, done, reward,violation_code,violation_string
        else:
                counter=len(self.triggers)
                i=0
                #print('duration',duration)
                while i <counter:  # update service time required
                    trigger = self.triggers[i]
                    if isinstance(trigger, EndService):
                        #print('self.services.loc[trigger.train,trigger.service]__before',
                                  #self.services.loc[trigger.train,trigger.service])
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
                            #print('end_servise_from_next_state:')
                            self._end_service(i)
                            counter-=1
                            i-=1

                        #print('self.services.loc[trigger.train,trigger.service]__after',
                                  #self.services.loc[trigger.train, trigger.service])
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

                solved, done,violation_code_eval,violation_string_eval = self._evaluate()
                if solved:
                        reward+=5
                if violation_code==None:
                        violation_code=violation_code_eval
                        violation_string=violation_string_eval
             
        #done += fail_in_action
        self.state_order += 1
        # clear the action
        self.action = None
        #print('is next trigger Arrival: ',isinstance(self.triggers[0], Arrival))
        #print('is next trigger Departure: ',isinstance(self.triggers[0], Departure))
        #print('is next trigger End service: ',isinstance(self.triggers[0], EndService))
        #print('time to next triger: ',self.triggers[0].time)
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

    def _get_not_empty_tracks(self,report_error):
        not_empty_tracks_and_wait_possible=[]
        for i in self.tracks:
                not_empty_tracks_and_wait_possible+=[i]
        if  (self.triggers[0].time>0) or (isinstance(self.triggers[0], Departure)==0 and isinstance(self.triggers[0], Arrival)==0):
        #if (isinstance(self.triggers[0], Arrival)==0 or self.triggers[0].time>0) and (isinstance(self.triggers[0], Departure)==0 or self.triggers[0].time>0):
                not_empty_tracks_and_wait_possible+=[100]
        if (isinstance(self.triggers[0], Departure)==1 and self.triggers[0].time==0):
                not_empty_tracks_and_wait_possible+=['d']
        right=0
        if report_error==1:
                if 'd' in not_empty_tracks_and_wait_possible:
                    right+=1
                if 16 in not_empty_tracks_and_wait_possible:
                    right+=1
                if 100 in not_empty_tracks_and_wait_possible:
                    right+=1
                if right!=1:
                        print('!!!!!',not_empty_tracks_and_wait_possible)
                        print('Arrival',isinstance(self.triggers[0], Arrival))
                        print('Departure',isinstance(self.triggers[0], Departure))
                        print(self.triggers[0].time)
        return not_empty_tracks_and_wait_possible

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
        return num_dep, num_train_yrd, num_train_s_track, next_trigger



    def _evaluate(self):  # return solved, done
        violation_code,violation_string=None,None
        if self.triggers == []:
            return 1, 1,violation_code,violation_string  # solved
        else:
            for t in self.triggers:
                if t.time < 0:
                    if isinstance(self.triggers[0], Arrival):
                        #print('missed arrival---------------------------------------')
                        violation_code,violation_string=1009,'missed_arrival'
                    if isinstance(self.triggers[0], Departure):
                        #print('missed departure----------------------------------------')
                        violation_code,violation_string=1010,'missed_departure'

                    return 0, 1,violation_code,violation_string  # failed

            return 0, 0,violation_code,violation_string  # not terminal

    def _backup_yrd_info(self):
        return deepcopy(self)

    def _restore_yrd(self,backup_yard):
         self=deepcopy(backup_yard)
         return None




