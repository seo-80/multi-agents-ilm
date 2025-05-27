import numpy as np
import tqdm
import itertools
from . import agents, networks, recorders, markov_chain



def simulate(
    args,
):
    simulation_count = args.get("simulation_count")
    agent = args.get("agent")
    agents_count = args.get("agents_count")
    agents_arguments = args.get("agents_arguments")
    network = args.get("network")
    network_args = args.get("network_args")
    recorder = args.get("recorder")
    simulate_type = args.get("simulate_type")
    initial_states = args.get("initial_states")

    if "nonzero_alpha" in agents_arguments.keys():
        temp_arts = agents_arguments.copy()
        if agents_arguments["nonzero_alpha"] == "evely":
            if 'Infinite' in agent:
                temp_agents_arguments = [{"alpha": agents_arguments["alpha"], "data_size": agents_arguments["data_size"], "agentgent_number":i} for i in range(agents_count)]
            else:
                temp_agents_arguments = [{"alpha": agents_arguments["alpha"], "data_size": agents_arguments["data_size"], "variants_count": agents_arguments["variants_count"]} for _ in range(agents_count)]
        elif agents_arguments["nonzero_alpha"] == "center":
            if 'Infinite' in agent:
                temp_agents_arguments = [{"alpha": 0, "data_size": agents_arguments["data_size"], "agentgent_number":i} for i in range(agents_count)]
                temp_agents_arguments[agents_count // 2]["alpha"] = agents_arguments["alpha"]
            else:
                temp_agents_arguments = [{"alpha": 0, "data_size": agents_arguments["data_size"], "variants_count": agents_arguments["variants_count"]} for _ in range(agents_count)]
                temp_agents_arguments[agents_count // 2]["alpha"] = agents_arguments["alpha"]
        else:
            raise ValueError("nonzero_alpha must be 'evely' or 'center'")
        agents_arguments = temp_agents_arguments


    if simulate_type == "markov_chain":
        return simulate_markov_chain(
            simulation_count=simulation_count,
            agent=agent,
            agents_count=agents_count,
            agents_arguments=agents_arguments,
            network=network,
            network_args=network_args,
            recorder=recorder,
            initial_states=initial_states,
        )
    if type(agent) == str:
        agent = agents.agent(agent)
    if agents_count is None:
        agents_count=1

    #init network
    if network is None:
        network = networks.network(agents_count=agents_count, args=network_args)

    #init recorder
    if type(recorder) == str:
        recorder=recorders.recorder(recorder_type=recorder, simulation_count=simulation_count)
    
    #init agents
    if type(agents_arguments) == dict:
        agents_arguments=[agents_arguments for _ in range(agents_count)]
    agents_list = [agent(**arg) for arg in agents_arguments]
    total_data_counts=[agent.data_size for agent in agents_list]
    for si in tqdm.trange(simulation_count):
        data_flow_count=networks.generate_data_flow_count(network,total_data_counts=total_data_counts)
        datas=generate_datas(agents_list=agents_list,data_flow_counts=data_flow_count)
        for ai, agent in enumerate(agents_list):
            agent.learn(datas[ai])
        recorder(agents=agents_list)
    return recorder

def simulate_markov_chain(
    simulation_count,
    agent,
    agents_count=None,
    agents_arguments=None,
    network=None,
    network_args=None,
    recorder="data",
    initial_states=None,
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
    states = np.zeros([agents_arguments[i]["data_size"] + 1 for i in range(agents_count)])
    if initial_states is None:
        if agent == "BayesianFiniteVariantsAgent":
            init_state = np.zeros(agents_count, dtype=int)
            states[tuple(init_state)] = 1
            # for ai in range(agents_count):
            #     init_state = np.zeros(agents_count, dtype=int)
            #     init_state[ai] = 1
            #     states[tuple(init_state)] = 1/agents_count
        elif agent == "BayesianInfiniteVariantsAgent":
            new_variant_probability = markov_chain.get_new_variant_probability(
                agents_arguments=agents_arguments,
                network=network 
            )
            for ai in range(agents_count):
                init_state = np.zeros(agents_count, dtype=int)
                init_state[ai] = 1
                states[tuple(init_state)] = new_variant_probability[ai]
    else:
        if len(initial_states) == agents_count:
            states[tuple(initial_states)] = 1
        if initial_states.shape == tuple([agents_arguments[i]["data_size"] + 1 for i in range(agents_count)]):
            states = initial_states

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
    

def generate_datas(agents_list, data_flow_counts):  # todo generalize this function
    if type(agents_list[0]) == agents.BayesianFiniteVariantsAgent:
        datas = np.zeros((len(agents_list), agents_list[0].variants_count))
        for i, _ in enumerate(agents_list):
            for j, learner in enumerate(agents_list):
                datas[i] += agents_list[j].produce(n=data_flow_counts[i][j])
        datas = np.zeros((len(agents_list), agents_list[0].variants_count))
        for i, _ in enumerate(agents_list):
            for j, learner in enumerate(agents_list):
                datas[i] += agents_list[j].produce(n=data_flow_counts[i][j])
        return datas
    elif type(agents_list[0]) == agents.BayesianInfiniteVariantsAgent:
        datas = [None for _ in agents_list]
        for i, _ in enumerate(agents_list):
            for j, learner in enumerate(agents_list):
                if data_flow_counts[i][j] > 0:
                    if datas[i] is None:
                        datas[i] = agents_list[j].produce(n=data_flow_counts[i][j])
                    else:
                        datas[i] = np.concatenate((datas[i], agents_list[j].produce(n=data_flow_counts[i][j])))
        return np.array(datas)

