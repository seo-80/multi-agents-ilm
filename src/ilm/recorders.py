import numpy
import itertools
import collections


def recorder(recorder_type=None, simulation_count=None):
    if recorder_type is None:
        return Recorder(simulation_count=simulation_count)
    elif recorder_type == "data":
        return DataRecorder(simulation_count=simulation_count)
    elif recorder_type == "distance":
        return DistanceRecorder(simulation_count=simulation_count)
    elif recorder_type == "data_state_vec":
        return DataRecorderStateVec(simulation_count=simulation_count)
    else:
        raise ValueError(f"Unknown recorder type: {recorder_type}")


class Recorder:
    def __init__(self,simulation_count,data_shape=None, simularion_args = None):
        self.__simulation_count = simulation_count
        # if not simulation_count is None and not data_shape is None:
        #     self.__return_record = numpy.empty(((simulation_count,)+ data_shape))
        # else:
        self.__return_record = None
        self.__i=0
        self.__simularion_args = simularion_args
        self.__agent_type = None
        
    def __call__(self,**kwargs):
        if self.__agent_type is None:
            self.__agent_type = kwargs["agents"][0].__class__.__name__
        tmp_record=self.fileter_function(**kwargs)
        if self.__return_record is None:
            self.__return_record=numpy.empty((self.__simulation_count,)+tmp_record.shape)
        self.__return_record[self.__i]=tmp_record
        self.__i+=1
    
    #そのまま返すとreturn_recordの初期化でエラー
    # def fileter_function(self,**kwargs):
    #     return **kwargs
    @property
    def simularion_args(self):
        return self.__simularion_args
    @property
    def record(self):
        return self.__return_record

class DataRecorder(Recorder):
    def __init__(self,simulation_count,data_shape=None):
        super().__init__(simulation_count=simulation_count,data_shape=data_shape)
        self.__distance = None
        self.__oldness = None
        self.__expected_distance = None
        self.__expected_oldness = None
    def compute_oldness(self):
        if self._Recorder__agent_type == "BayesianInfiniteVariantsAgent":
            self.__oldness = numpy.array([[numpy.mean([t-d[0] for d in agent]) for agent in self._Recorder__return_record[t]] for t in range(self._Recorder__simulation_count)])
            self.__expected_oldness = numpy.mean(self.__oldness[self.simulation_count//10:], axis=0)
        elif self._Recorder__agent_type == "BayesianFiniteVariantsAgent":
            print('cant compute oldness for BayesianFiniteVariantsAgent')

    def keys(self):
        return ["oldness","distance","expected_distance","record"]
    
    def __getitem__(self,key):
        if key == "oldness":
            return self.oldness
        elif key == "distance":
            return self.distance
        elif key == "expected_distance":
            return self.expected_distance
        else:
            raise ValueError(f"Unknown key: {key}")

    @property
    def oldness(self):
        if self.__oldness is None:
            self.compute_oldness()
        return self.__oldness
    
    def fileter_function(self,**kwargs):
        return numpy.array([agent.data for agent in kwargs["agents"]])
    @property
    def expected_distance(self):
        if self.__expected_distance is None:
            self.compute_distance()
        return self.__expected_distance
        

    @property
    def distance(self):
        if self.__distance is None:
            self.compute_distance()
        return self.__distance
    
    @property
    def simulation_count(self):
        return self._Recorder__simulation_count
    def compute_distance(self):
        if self._Recorder__agent_type == "BayesianInfiniteVariantsAgent":
            self.compute_oldness()
            collections_rec = [[collections.Counter([tuple(d) for d in agent]) for agent in self._Recorder__return_record[t]] for t in range(self._Recorder__simulation_count)]
            temp_distance = numpy.array([[[sum(abs(agenti[k]-agentj[k]) for k in agenti.keys()) for agenti in collections_rec[t]] for agentj in collections_rec[t]] for t in range(self._Recorder__simulation_count)])
            agents_count = temp_distance.shape[1]
            self.__distance = numpy.array([[[temp_distance[(t, i,j)] + temp_distance[(t, j, i)] for i in range(agents_count)] for j in range(agents_count)] for t in range(self._Recorder__simulation_count)])
            self.__expected_distance = numpy.sum(self.__distance, axis=0)
        elif self._Recorder__agent_type == "BayesianFiniteVariantsAgent":
            self.__distance = numpy.array([[[numpy.abs(agenti-agentj).sum() for agenti in self._Recorder__return_record[t]] for agentj in self._Recorder__return_record[t]] for t in range(self._Recorder__simulation_count)])
            self.__expected_distance = numpy.mean(self.__distance[self.simulation_count//10:], axis=0)            
class DataRecorderStateVec(Recorder):
    def __init__(self,simulation_count,data_shape=None, distances_matrix=None):
        super().__init__(simulation_count=simulation_count,data_shape=data_shape)
        self.distances_matrix = distances_matrixs
        self.__distance = None
        self.__expected_distance = None
    
    def fileter_function(self,**kwargs):
        return numpy.array(kwargs["states"])
    


    def compute_distance(self):
        agents_count = len(self._Recorder__return_record.shape) - 1
        if self.__distance is None:
            if self.distances_matrix is None:
                self.distances_matrix = numpy.empty((agents_count,) * 2 + self._Recorder__return_record.shape[1:])
                for index in itertools.product(*[range(ds) for ds in self.distances_matrix.shape]):
                    self.distances_matrix[index] = abs(index[index[0] + 2] - index[index[1] + 2])
            self.__distance = numpy.tensordot(self._Recorder__return_record, self.distances_matrix, axes=(range(1, agents_count + 1), range(2, agents_count + 2)))
        if self.__expected_distance is None:
            self.__expected_distance = numpy.tensordot(self._Recorder__return_record, self.distances_matrix, axes=(range(1, agents_count + 1), range(2, agents_count + 2)))
    @property
    def distance(self):
        if self.__distance is None:
            self.compute_distance()
        return self.__distance
        # return numpy.tensordot( self.distances_matrix,self._Recorder__return_record, axes=(range(2, agents_count + 2), range(1, agents_count + 1)))

        # distances_record =numpy.tensordot(states_record, distances_matrix, axes=(range(1, agents_count + 1), range(2, agents_count + 2)))

class DistanceRecorder(Recorder):
    def __init__(self,simulation_count,data_shape=None):
        super().__init__(simulation_count=simulation_count,data_shape=data_shape)
    
    def fileter_function(self,**kwargs):
        return numpy.array([[numpy.abs(agenti.data-agentj.data).sum() for agenti in kwargs["agents"]] for agentj in kwargs["agents"]])

class FilterFunction:#argsは(data,agents)
    def filter_data(**kwargs):
        return kwargs["a"]
        
