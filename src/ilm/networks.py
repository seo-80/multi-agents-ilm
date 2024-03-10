import numpy

DEFAULT_FLOW_RATE=0.01

def network(network_type, agents_count, args=None):
    return_network=numpy.identity(agents_count)
    if network_type == "outer":
        if args is None:
            fr=DEFAULT_FLOW_RATE
        else:
            fr=arg["flow_rate"]



    return return_network

def generate_data_flow_count(data_flow_rate,total_data_count=None,total_data_counts=None):
    if total_data_counts is None:
        if total_data_count is None:
            raise ValueError("Either total_data_count orntotal_data_counts is required")
        total_data_counts=[total_data_count for _ in data_flow_rate]
    data_flow_count=numpy.empty_like(data_flow_rate)
    for i, rate in enumerate(data_flow_rate):
        data_flow_count[i]=numpy.random.multinomial(n=total_data_counts[i],pvals=rate)
    print(data_flow_count)
    return data_flow_count
