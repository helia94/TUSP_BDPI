import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gym_envs.global_const import *
from yrd_main import *
import time

#rescale time linearly
def time_transition_trigger(x):
    y=x/(12*60)
    #y=np.exp(-1*self.exp*x)
    return y

#rescale time linearly
def time_transition_task(x):
    y=x/(60)
    #y=np.exp(-1*self.exp*x)
    return y

class Map(object):
    def __init__(self, train_num, track_num):
        self.train_num = train_num
        self.track_num = track_num
##        #for none linear transition in time
##        self.exp = exp 



    def _rep_status(self, state):
        # for each train, current position will be 1
        d={}
        for i in range(5):
            d["s_map{0}".format(i+1)] = np.zeros([self.train_num, self.track_num])
        for track in state.tracks:
            for i in range(len(state.tracks[track])):
                train=state.tracks[track][i]
                d["s_map{0}" .format(i+1)][state.train_idx[train.train], track] = 1
        status_map_p1=d['s_map1']
        status_map_p2=d['s_map2']
        status_map_p3=d['s_map3']
        status_map_p4=d['s_map4']
        status_map_p5=d['s_map5']
        return status_map_p1, status_map_p2, status_map_p3, status_map_p4, status_map_p5




    def _rep_length(self, state):
        # for each train, current position will be will be length
        train_length_vec=np.zeros([self.train_num, 1])
        for track in state.tracks:
            for train in state.tracks[track]:
                train_length_vec[state.train_idx[train.train], 0] = train.length/500
        for i in range(len(state.not_arrived)):
            train_length_vec[state.train_idx[state.not_arrived[i].train], 0] = state.not_arrived[i].length/500
        return train_length_vec

    def _rep_arrival_binary(self, state):
        # for each train, will be 1 if its time for its arrival
         arrival_binary_vec= np.zeros([self.train_num, 1])
         if GATE_TRACK in state.tracks:
            train=state.tracks[GATE_TRACK][0]
            arrival_binary_vec[state.train_idx[train.train], 0] =1
         return arrival_binary_vec


    def _rep_service(self, state):
        # for each train, and each service service, the the value of required time scaled to [0,1]
        service_map = np.zeros([len(list(state.services)), self.train_num, 1])
        for train in state.services.index:
            for service in list(state.services):
                req_time=state.services.loc[train, service]
                service_map[service-1, state.train_idx[train], 0] = time_transition_task(req_time)
        return service_map

    def _rep_inservice(self, state):
        # for each train, if it is currently in service, it will be one
        inservice_map = np.zeros([self.train_num, 1])
        for trigger in state.triggers:
            if isinstance(trigger, EndService):
                for track in state.tracks:
                    for train in state.tracks[track]:
                        if trigger.train==train.train:
                            inservice_map[state.train_idx[trigger.train], 0] = 1
        return inservice_map

    def _rep_arrival(self, state):
        # for each unarrived train, time to arrival scaled to [0,1]
        arrival_vec = np.zeros([self.train_num, 1])
        for trigger in state.triggers:
            if isinstance(trigger, Arrival):
                time_to_arrival=trigger.time
                arrival_vec[state.train_idx[trigger.train], 0] = time_transition_trigger(time_to_arrival)
        return arrival_vec

    def _rep_departure(self, state):
        # for each train get its earliest/latest possible depart time scaled to [0,1]
        # and whether it is possible to depart now in binary

        # get material-train matching
        material_train_dic = {}
        for train in state.not_arrived:
            if train.material not in material_train_dic:
                material_train_dic[train.material] = []
            material_train_dic[train.material] += [train.train]
        for track in state.tracks:
            for train in state.tracks[track]:
                if train.material not in material_train_dic:
                    material_train_dic[train.material] = []
                material_train_dic[train.material] += [train.train]

        # get train-depart time matching
        train_dtime_dic = {}
        for trigger in state.triggers:
            if isinstance(trigger, Departure):
                for train in material_train_dic[trigger.material]:
                    if train not in train_dtime_dic:
                        train_dtime_dic[train] = []
                    train_dtime_dic[train] += [trigger.time]

        # construct map
        first_departure_vec = np.zeros([self.train_num, 1])
        second_departure_vec = np.zeros([self.train_num, 1])
        third_departure_vec = np.zeros([self.train_num, 1])
##        forth_departure_vec = np.zeros([self.train_num, 1])
        late_departure_vec = np.zeros([self.train_num, 1])
        departure_binary_vec=np.zeros([self.train_num, 1])
        for train in train_dtime_dic:
            dep_time_list=list(train_dtime_dic[train])
            dep_time_list.sort()
            n_list=len(dep_time_list)
            first_departure_vec[state.train_idx[train], 0] = time_transition_trigger(dep_time_list[0])
            second_departure_vec[state.train_idx[train], 0] = time_transition_trigger(dep_time_list[min(1,n_list-1)])
            third_departure_vec[state.train_idx[train], 0] = time_transition_trigger(dep_time_list[min(2,n_list-1)])
##          forth_departure_vec[state.train_idx[train], 0] = time_transition_trigger(dep_time_list[min(3,n_list-1)])
            late_departure_vec[state.train_idx[train], 0] = time_transition_trigger(dep_time_list[n_list-1])
            if dep_time_list[0]==0:
                departure_binary_vec[state.train_idx[train], 0]=1
        return first_departure_vec,second_departure_vec,third_departure_vec,late_departure_vec,departure_binary_vec


    #concat all information on trains that are independent of location on tracks
    def _write_train_spec_features(self,state):
        train_length_vec =self._rep_length(state)
        arrival_vec = self._rep_arrival(state)
        first_departure_vec,second_departure_vec,third_departure_vec,late_departure_vec,departure_binary_vec = self._rep_departure(state)
        arrival_binary_vec=self._rep_arrival_binary(state)
        service_vec=self._rep_service(state)
        inservice_vec =self._rep_inservice(state)

        list_vec=[arrival_binary_vec,train_length_vec,arrival_vec,first_departure_vec,second_departure_vec,
                  third_departure_vec,late_departure_vec,departure_binary_vec,inservice_vec,service_vec]
        list_map=['arrival_binary_map','train_length_map','arrival_map','first_departure_map',
                  'second_departure_map','third_departure_map','late_departure_map',
                  'departure_binary_map','inservice_map','service']
        n=len(list_map)
        map_dic={}
        for i in range(n):
            if len(np.shape(list_vec[i]))>2:
                for j in range (np.shape(list_vec[i])[0]):
                    map_dic[list_map[i]+'_'+str(j)]=list_vec[i][j]
            else:
                map_dic[list_map[i]]=list_vec[i]

        map_dic_head=list(map_dic.keys())
        return map_dic, map_dic_head

    #construct the final vector  
    def rep(self, state):
        self.train_num= TRAIN_NUM  
        self.track_num= TRACK_NUM  
        status_map_p1, status_map_p2, status_map_p3, status_map_p4 , status_map_p5 = self._rep_status(state)
        map_dic, map_dic_head= self._write_train_spec_features(state)
        status_maps = np.stack([status_map_p1, status_map_p2, status_map_p3, status_map_p4, status_map_p5], axis=2)
        status_maps= np.delete(status_maps, [0,1,10,14,15,17], axis=1)
        
        for name in map_dic_head:
            if 'aggregated_map' in locals():
                aggregated_map=np.append(aggregated_map,map_dic[name], axis=0)
            else:
                aggregated_map=map_dic[name]
        aggregated_map=np.concatenate([aggregated_map.flatten(), status_maps.flatten()])
        return aggregated_map

