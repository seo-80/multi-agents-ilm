import numpy as np
import ilm
import tqdm




def possible_states(
    agents_arguments,
    agents_count=None
):
    if agents_count is None:
        agents_count=len(agents_arguments)
    data_sizes=np.array([arg["data_size"] for arg in agents_arguments])
    ret_states=np.empty((np.prod(data_sizes+1, axis=0), agents_count), dtype=int)
    tmp_state=np.zeros(agents_count)
    for i in range(len(ret_states)):
        for ai in range(agents_count):
            if tmp_state[ai] > data_sizes[ai]:
                tmp_state[ai] = 0
                tmp_state[ai+1] +=1
        ret_states[i]=tmp_state
        tmp_state[0]+=1
    return ret_states

def binomial(pval, x_v, M):
    C=np.math.factorial(M)/(np.math.factorial(x_v)*np.math.factorial(M-x_v))
    prob=C*(pval**x_v*(1-pval)**(M-x_v))
    return prob

def transition_probability(prestate, poststate, network, agents_arguments):
    agents_count=len(prestate)
    alphas=np.array([arg["alpha"] for arg in agents_arguments])
    data_sizes=np.array([arg["data_size"] for arg in agents_arguments], dtype=int)
    valiation_probability=np.array([np.dot(prestate/data_sizes*(1-alphas), network[i]) for i in range(agents_count)])
    ret_probability=np.prod([binomial(valiation_probability[ai], poststate[ai], data_sizes[ai]) for ai in range(agents_count)], axis=0)
    return ret_probability

def transition_matrix(
    agents_arguments,
    agents_count=None,
    network=None,
):
    if agents_count is None:
        if not type(agents_arguments) == list:
            raise ValueError
        agents_count=len(agents_arguments)

    #init network
    if type(network) == str or network is None:
        network = ilm.networks.network(network_type=network,agents_count=agents_count)
    
    states=possible_states(agents_arguments)
    transition_matrix=np.array([
        [transition_probability(prestate, poststate, network, agents_arguments) for poststate in states] for prestate in states
    ])
    return transition_matrix

    
    

if __name__ == "__main__":#テスト
    network=np.array([
        [0.8,0.1,0.1],
        [0,1,0],
        [0.1,0.1,0.8],
    ])
    # network=np.array([
    #     [1,0,0],
    #     [0,1,0],
    #     [0,0,1],
    # ])
    network=np.identity(5)
    agents_arguments=[
        {"alpha":0.,"data_size":1},
        {"alpha":0.,"data_size":1},
        {"alpha":0.01,"data_size":1},
        {"alpha":0.,"data_size":1},
        {"alpha":0.,"data_size":1}
    ]
    m=transition_matrix(
        agents_arguments=agents_arguments,
        network="outer"
    )
    data_sizes=np.array([arg["data_size"] for arg in agents_arguments], dtype=int)

    import matplotlib.pyplot as plt
    simulation_count=100
    agents_count=len(agents_arguments)
    states_list=possible_states(agents_arguments=agents_arguments)
    init_state=np.zeros(agents_count);init_state[agents_count//2]=1
    print(init_state)
    states=np.array(np.all(states_list==init_state, axis=1),dtype=float)
    distances=np.empty((states_list.shape[0], agents_count*(agents_count-1)//2))
    for si, state in enumerate(states_list):
        for ai in range(agents_count):
            for aj in range(ai+1,agents_count):
                distances[si,ai+aj-1]=np.abs(state[ai]-state[aj])
    for i in range(states_list.shape[0]):
        print(states_list[i],distances[i])
    print(distances.T@states)
    distances_record=np.empty((simulation_count, agents_count*(agents_count-1)//2))
    rai=1
    one_agents_record=np.empty((simulation_count,data_sizes[rai]+1))
    one_agent_matrix=np.array([list(map(lambda state: int(state[rai]==vc),  states_list)) for vc in range(data_sizes[rai]+1)])
    print("simulating....")
    for i in tqdm.tqdm(range(simulation_count)):
        one_agents_record[i]=one_agent_matrix@states
        distances_record[i]=distances.T@states
        states=m.T@states
    
    fig, ax=plt.subplots()
    ax.plot(distances_record)
    print(agents_count)
    ax.legend([str(ai)+str(aj) for ai in range(agents_count) for aj in range(ai+1,agents_count)])
    # ax.pcolor(one_agents_record.T)
    plt.show()


