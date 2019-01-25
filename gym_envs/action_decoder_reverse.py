
def label_decoder_reverse(not_empty_tracks_and_wait_possible):
    actions=[]
    #add wait if wait is possible
    if 100 in not_empty_tracks_and_wait_possible:
        actions+=[53]
    #add all actions from parking and cleaning tracks that are empty
    for i in [2,3,4,5]:
        if i in not_empty_tracks_and_wait_possible:
            actions+=[9+i-2,33+i-2]
    for i in [6,7,8,9]:
        if i in not_empty_tracks_and_wait_possible:
            actions+=[9+i-2,33+i-2,17+i-6,17+4+i-6]
    for i in [11,12]:
        if i in not_empty_tracks_and_wait_possible:
            actions+=[25-11+i,27-11+i,29-11+i,31-11+i]+[49-11+i]

    #if train on gate track then only allow parking move from gate track       
    for i in [16]:
        if i in not_empty_tracks_and_wait_possible:
            actions=list(range(1,9))
    #if train on relocation track then only allow move from relocation track 
    for i in [13]:
        if i in not_empty_tracks_and_wait_possible:
            actions=list(range(41,49))+[51,52]
    #if time for departure then only allow departure movements
    if 'd' in not_empty_tracks_and_wait_possible:
        actions=list(set(actions).intersection(list(range(9,17))))
    if 'w' in not_empty_tracks_and_wait_possible:
        actions=[53]
    #if not time for departure then remove departure movements
    else:
        actions=list(set(actions).intersection(list(range(1,9))+list(range(17,54))))
    actions=[x-1 for x in actions]
    return actions 

    
