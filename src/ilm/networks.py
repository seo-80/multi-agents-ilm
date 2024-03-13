import numpy

DEFAULT_FLOW_RATE=0.01

def network(agents_count, args=None):
    return_network=numpy.identity(agents_count)
    if not args is None:
        if "outer_flow_rate" in args:
            outer_frow_rate=args["outer_flow_rate"]
            if "center_index" in args:
                center_index = args["center_index"]
            else:
                center_index=agents_count//2
            for ai in range(agents_count):
                if ai < center_index:
                    return_network[ai][ai+1]+=outer_frow_rate
                    return_network[ai][ai]-=outer_frow_rate
                elif ai > center_index:
                    return_network[ai][ai-1]+=outer_frow_rate
                    return_network[ai][ai]-=outer_frow_rate



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
