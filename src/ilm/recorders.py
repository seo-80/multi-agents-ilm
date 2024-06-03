import numpy
import itertools


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
        if not simulation_count is None and not data_shape is None:
            self.__return_record = numpy.empty(((simulation_count,)+ data_shape))
        else:
            self.__return_record = None
        self.__i=0
        self.__simularion_args = simularion_args
        
    def __call__(self,**kwargs):
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
    
    def fileter_function(self,**kwargs):
        return numpy.array([agent.data for agent in kwargs["agents"]])
    
    @property
    def distance(self):
        if self.__distance is None:
            self.__distance = numpy.array([[[numpy.abs(agenti-agentj).sum() for agenti in self._Recorder__return_record[t]] for agentj in self._Recorder__return_record[t]] for t in range(self._Recorder__simulation_count)])
        return self.__distance
    
    @property
    def simulation_count(self):
        return self._Recorder__simulation_count
    def compute_distance(self,**kwargs):
        return numpy.array([[numpy.abs(agenti.data-agentj.data).sum() for agenti in kwargs["agents"]] for agentj in kwargs["agents"]])

class DataRecorderStateVec(Recorder):
    def __init__(self,simulation_count,data_shape=None, distances_matrix=None):
        super().__init__(simulation_count=simulation_count,data_shape=data_shape)
        self.distances_matrix = distances_matrix
        self.__distance = None
    
    def fileter_function(self,**kwargs):
        return numpy.array(kwargs["states"])
    

    
    @property
    def distance(self):
        if self.__distance is None:
            agents_count = len(self._Recorder__return_record.shape) - 1
            if self.distances_matrix is None:
                self.distances_matrix = numpy.empty((agents_count,) * 2 + self._Recorder__return_record.shape[1:])
                for index in itertools.product(*[range(ds) for ds in self.distances_matrix.shape]):
                    self.distances_matrix[index] = abs(index[index[0] + 2] - index[index[1] + 2])
            self.__distance = numpy.tensordot(self.distances_matrix,self._Recorder__return_record, axes=(range(2, agents_count + 2), range(1, agents_count + 1)))
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
        
