import numpy as np
import tqdm
import itertools
from . import agents, networks, recorders, markov_chain



def simulate(
    simulation_count,
    agent,
    agents_count=None,
    agents_arguments=None,
    network=None,
    network_args=None,
    recorder="data",
    simulate_type="markov_chain"
):
    if simulate_type == "markov_chain":
        return simulate_markov_chain(
            simulation_count=simulation_count,
            agent=agent,
            agents_count=agents_count,
            agents_arguments=agents_arguments,
            network=network,
            network_args=network_args,
            recorder=recorder
        )
    if type(agent) == str:
        agent = agents.agent(agent)
    if agents_count is None:
        agents_count=1

    #init network
    if type(network) == dict or network is None:
        network = networks.network(agents_count=agents_count,args=network_args)

    #init recorder
    if type(recorder) == str:
        recorder=recorders.recorder(recorder_type=recorder, simulation_count=simulation_count)
    
    #init agents
    if type(agents_arguments) == dict:
        agents_arguments=[agents_arguments for _ in range(agents_count)]
    agents_list = [agent(**arg) for arg in agents_arguments]
    total_data_counts=[agent.data_size for agent in agents_list]
    for si in tqdm.trange(simulation_count):
        recorder(agents=agents_list)
        data_flow_count=networks.generate_data_flow_count(network,total_data_counts=total_data_counts)
        datas=generate_datas(agents=agents_list,data_flow_counts=data_flow_count)
        for ai, agent in enumerate(agents_list):
            agent.learn(datas[ai])
    return recorder

def simulate_markov_chain(
    simulation_count,
    agent,
    agents_count=None,
    agents_arguments=None,
    network=None,
    network_args=None,
    recorder="data",
):
    if not type(agent) == str:
        raise ValueError("agent must be a string if simulate_type is 'markov_chain'")
    if agents_count is None:
        agents_count = len(agents_arguments)

    if network is None:
        network = networks.network(agents_count=agents_count, args=network_args)
    
    if type(recorder) == str:
        recorder = recorders.recorder(recorder_type=recorder + "_state_vec", simulation_count=simulation_count)

    if type(agents_arguments) == dict:
        agents_arguments = [agents_arguments for _ in range(agents_count)]

    m = markov_chain.transition_matrix(
        agents_arguments=agents_arguments,
        agent_type=agent,
        network=network,
    )
    print("shape of m", m.shape)
        

    states = np.zeros([agents_arguments[i]["data_size"] + 1 for i in range(agents_count)])
    if agent == "BayesianFiniteVariantsAgent":
        for ai in range(agents_count):
            init_state = np.zeros(agents_count, dtype=int)
            init_state[ai] = 1
            states[tuple(init_state)] = 1/agents_count
    elif agent == "BayesianInfiniteVariantsAgent":
        new_variant_probability = markov_chain.get_new_variant_probability(
            agents_arguments=agents_arguments,
            network=network 
        )
        for ai in range(agents_count):
            init_state = np.zeros(agents_count, dtype=int)
            init_state[ai] = 1
            states[tuple(init_state)] = new_variant_probability[ai]
    print("sum states", np.sum(states))
    

    for i in tqdm.tqdm(range(simulation_count)):
        recorder(states=states)
        states = np.tensordot(m, states, axes=(range(agents_count, agents_count * 2), range(agents_count)))

    if recorder == "variants_frequency":
        for ai in range(agents_count):
            vf = np.tensordot(states_record.sum(axis=tuple(range(1, ai + 1)) + tuple(range(ai + 2, agents_count))), [0, 1], axes=((1), (0)))
            
        return vf
    elif recorder == "distance":
        distances_matrix = np.empty((agents_count,) * 2 + states.shape)
        for index in itertools.product(*[range(ds) for ds in distances_matrix.shape]):
            distances_matrix[index] = abs(index[index[0] + 2] - index[index[1] + 2])

        distances_record = np.tensordot(states_record, distances_matrix, axes=(range(1, agents_count + 1), range(2, agents_count + 2)))
        return distances_record
    return recorder
    

def generate_datas(agents,data_flow_counts):
    datas=np.zeros((len(agents),agents[0].variants_count))
    for i,_ in enumerate(agents):
        for j,learner in enumerate(agents):
            datas[i]+=agents[j].produce(n=data_flow_counts[i][j])
    return datas
        
        