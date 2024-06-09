import numpy

DEFAULT_FLOW_RATE=0.01

def network(agents_count, args=None):
    return_network=numpy.identity(agents_count)
    if not args is None:
        if "center_index" in args:
            center_index = args["center_index"]
        else:
            center_index=agents_count//2
        if "outward_flow_rate" in args:
            outward_frow_rate=args["outward_flow_rate"]
            for ai in range(agents_count):
                if ai < center_index:
                    return_network[ai][ai+1]+=outward_frow_rate
                    return_network[ai][ai]-=outward_frow_rate
                elif ai > center_index:
                    return_network[ai][ai-1]+=outward_frow_rate
                    return_network[ai][ai]-=outward_frow_rate
        
        if "bidirectional_flow_rate" in args:
            bidirectional_flow_rate=args["bidirectional_flow_rate"]
            is_torus=args.get("is_torus",None)
            for ai in range(agents_count):
                return_network[ai][ai-1]+=bidirectional_flow_rate/2
                return_network[ai][(ai+1)%agents_count]+=bidirectional_flow_rate/2
                return_network[ai][ai]-=bidirectional_flow_rate

            if not is_torus:
                return_network[0][-1]-=bidirectional_flow_rate/2
                return_network[0][0]+=bidirectional_flow_rate/2
                return_network[-1][0]-=bidirectional_flow_rate/2
                return_network[-1][-1]+=bidirectional_flow_rate/2



    print(return_network)
    return return_network

def generate_data_flow_count(data_flow_rate,total_data_count=None,total_data_counts=None):
    if total_data_counts is None:
        if total_data_count is None:
            raise ValueError("Either total_data_count orntotal_data_counts is required")
        total_data_counts=[total_data_count for _ in data_flow_rate]
    data_flow_count=numpy.empty_like(data_flow_rate)
    for i, rate in enumerate(data_flow_rate):
        data_flow_count[i]=numpy.random.multinomial(n=total_data_counts[i],pvals=rate)  
    return data_flow_count
