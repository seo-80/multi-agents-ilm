import numpy as np
import tqdm
import itertools

import ilm
from operator import mul




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
    if type(agents_arguments) == dict:
        agents_arguments=[agents_arguments for _ in range(agents_count)]

    #init network
    if type(network) == dict or network is None:
        network = ilm.networks.network(agents_count=agents_count,args=network)
    
    states=possible_states(agents_arguments)
    transition_matrix=np.array([
        [transition_probability(prestate, poststate, network, agents_arguments) for prestate in states] for poststate in states
    ])
    transition_matrix=transition_matrix.reshape([arg["data_size"]+1 for arg in agents_arguments]*2)
    return transition_matrix
def transition_matrix(
    agents_arguments,
    agents_count=None,
    network=None,
):
    if agents_count is None:
        if not type(agents_arguments) == list:
            raise ValueError
        agents_count=len(agents_arguments)
    if type(agents_arguments) == dict:
        agents_arguments=[agents_arguments for _ in range(agents_count)]
    #init network
    if type(network) == dict or network is None:
        network = ilm.networks.network(agents_count=agents_count,args=network)

    transition_matrix=np.empty([arg["data_size"]+1 for arg in agents_arguments]*2)
    for prestate in (itertools.product(*[range(arg["data_size"]+1) for arg in agents_arguments])):
        for poststate in itertools.product(*[range(arg["data_size"]+1) for arg in agents_arguments]):
            transition_matrix[poststate+prestate]=transition_probability(prestate, poststate, network, agents_arguments)

    return transition_matrix


def get_new_variant_probability(
    agents_arguments,
    agents_count=None,
    network=None,
):
    if agents_count is None:
        if not type(agents_arguments) == list:
            raise ValueError
        agents_count=len(agents_arguments)
    if type(agents_arguments) == dict:
        agents_arguments=[agents_arguments for _ in range(agents_count)]
    #init network
    if type(network) == dict or network is None:
        network = ilm.networks.network(agents_count=agents_count,args=network)

    data_sizes=np.array([arg["data_size"] for arg in agents_arguments], dtype=int)
    alphas=np.array([arg["alpha"] for arg in agents_arguments], dtype=float)
    make_new_variant_probability=np.array(list(map(lambda a,n :a/(a+n), alphas, data_sizes)))
    return np.array(list(map(mul, data_sizes, network@make_new_variant_probability)))
    
    

if __name__ == "__main__":#テスト

    agents_arguments=[
        {"alpha":0.,"data_size":1},
        {"alpha":0.,"data_size":1},
        {"alpha":0.,"data_size":1},
        {"alpha":0.1,"data_size":1},
        {"alpha":0.,"data_size":1},
        {"alpha":0.,"data_size":1},
        {"alpha":0.,"data_size":1},
    ]



    agents_count=len(agents_arguments)
    network=ilm.networks.network(agents_count=agents_count,args={
        # "bidirectional_flow_rate":0.01,
        "outward_flow_rate":0.1,
        "is_torus":False
    })
    print(network)

    m=transition_matrix(
        agents_arguments=agents_arguments,
        network=network
    )
    new_variant_probability=get_new_variant_probability(
        agents_arguments=agents_arguments,
        network=network
    )
    print(new_variant_probability)
    data_sizes=np.array([arg["data_size"] for arg in agents_arguments], dtype=int)

    simulation_count=10000
    import matplotlib.pyplot as plt
    states=np.zeros([agents_arguments[i]["data_size"]+1 for i in range(agents_count)])
    for ai in range(agents_count):
        init_state=np.zeros(agents_count,dtype=int);init_state[ai]=1
        states[tuple(init_state)]=new_variant_probability[ai]
    print(new_variant_probability)
    rai=1
    print("simulating....")
    states_record=np.empty((simulation_count,)+states.shape)
    for i in tqdm.tqdm(range(simulation_count)):
        states_record[i]=states
        states = np.tensordot(m,states, axes=(range(agents_count,agents_count*2),range(agents_count)))


    what_plot="distances"
    if what_plot=="variants_frequency":
        fig, ax=plt.subplots()
        for ai in range(agents_count):
            plt_data=np.tensordot(states_record.sum(axis=tuple(range(1,ai+1))+tuple(range(ai+2,agents_count))), [0,1], axes=((1),(0)))
            ax.plot(plt_data)
        plt.show()
    if what_plot=="distances":#todo 異なるデータ数に拡張
        fig, ax=plt.subplots()
        distances_matrix=np.empty((agents_count,)*2+states.shape)
        for index in itertools.product(*[range(ds) for  ds in distances_matrix.shape]):
            distances_matrix[index]=abs(index[index[0]+2]-index[index[1]+2])

        distances_record=np.tensordot(states_record,distances_matrix,axes=(range(1,agents_count+1),range(2,agents_count+2)))
        # distances_record=np.zeros((simulation_count,)+(agents_count,)*2)
        # for t in range(simulation_count):
        #     for index in itertools.product(*[range(ds) for  ds in distances_matrix.shape]):
        #         distances_record[(t,)+(index[0],index[1])]+=states_record[(t,)+index[2:]]*abs(index[index[0]+2]-index[index[1]+2])



        legend=[]
        # for ai in itertools.combinations(range(agents_count),2):
        #     legend.append(str(ai[0])+str(ai[1]))
        #     ax.plot(distances_record[:,ai[0],ai[1]])
        ax.invert_xaxis()
        ax.pcolor(distances_record.sum(axis=0))
        ax.legend(legend)
        plt.show()
