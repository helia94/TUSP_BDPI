
from gym_envs.global_const import *
from yrd_main import *


def label_decoder(action_num):
    action_num=int(action_num)
    action_num+=1	
    if action_num==53:
        return Wait()
    elif action_num <=8:
        start_track=GATE_TRACK
        end_track=[2,3,4,5,6,7,8,9][action_num-1]
        return Move(None,None,None,start_track,end_track)
        #return action
    elif action_num <=8+8:
        start_track=[2,3,4,5,6,7,8,9][action_num-8-1]
        end_track=GATE_TRACK
        return Move(None,None,None,start_track,end_track)
        #return action
    elif action_num <=8+8+8:
        start_track=[6,7,8,9][(action_num-8-8-1)%4]
        end_track=[11,12][(action_num-8-8-1)//4]
        return Move(None,None,None,start_track,end_track)
        #return action
    elif action_num <=8+8+8+8:
        start_track=[11,12][(action_num-8-8-8-1)%2]
        end_track=[6,7,8,9][(action_num-8-8-8-1)//2]
        return Move(None,None,None,start_track,end_track)
        #return action
    elif action_num <=8+8+8+8+8:
        start_track=[2,3,4,5,6,7,8,9][(action_num-8-8-8-8-1)%8]
        end_track=13
        return Move(None,None,None,start_track,end_track)
        #return action
    elif action_num <=8+8+8+8+8+8:
        start_track=13
        end_track=[2,3,4,5,6,7,8,9][(action_num-8-8-8-8-8-1)%1]
        return Move(None,None,None,start_track,end_track)
        #return action
    elif action_num <=8+8+8+8+8+8+2:
        start_track=[11,12][(action_num-8-8-8-8-8-8-1)%2]
        end_track=13
        return Move(None,None,None,start_track,end_track)
        #return action
    else:
        start_track=13
        end_track=[11,12][(action_num-8-8-8-8-8-8-2-1)//2]
        return Move(None,None,None,start_track,end_track)
        #return action
