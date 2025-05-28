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
        if self.__agent_type is None and "agents" in kwargs:
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
        self.__variance_distance = None
        self.__variance_oldness = None

    def compute_oldness(self):
        if self._Recorder__agent_type == "BayesianInfiniteVariantsAgent":
            self.__oldness = numpy.array([[numpy.mean([t-d[0] for d in agent]) for agent in self._Recorder__return_record[t]] for t in range(self._Recorder__simulation_count)])
            self.__expected_oldness = numpy.mean(self.__oldness[self.simulation_count//10:], axis=0)
        elif self._Recorder__agent_type == "BayesianFiniteVariantsAgent":
            print('cant compute oldness for BayesianFiniteVariantsAgent')

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
    
    def compute_variance(self, compute_object = ["distance","oldness"]):
        if type(compute_object) == str:
            compute_object = [compute_object]
        agents_count = len(self._Recorder__return_record.shape) - 1
        if self.__variance_distance is None and "distance" in compute_object:
            self.__variance_distance = numpy.var(self.__distance, axis=0)
        if self.__variance_oldness is None and "oldness" in compute_object:
            self.__variance_oldness = numpy.var(self.__oldness, axis=0)
        
    def keys(self):
        return ["record","oldness","distance","expected_distance","expected_oldness","record","variance_distance","variance_oldness"]
    
    def __getitem__(self,key):
        if key == "oldness":
            return self.oldness
        elif key == "distance":
            return self.distance
        elif key == "expected_distance":
            return self.expected_distance
        elif key == "expected_oldness":
            return self.expected_oldness
        elif key == "record":
            return self.record
        elif key == "variance_distance":
            return self.__variance_distance
        elif key == "variance_oldness":
            return self.__variance_oldness
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
    def expected_oldness(self):
        if self.__expected_oldness is None:
            self.compute_oldness()
        return self.__expected_oldness
        

    @property
    def distance(self):
        if self.__distance is None:
            self.compute_distance()
        return self.__distance
    
    @property
    def simulation_count(self):
        return self._Recorder__simulation_count
    
    @property
    def variance_distance(self):
        if self.__variance_distance is None:
            self.compute_variance()
        return self.__variance_distance
    @property
    def variance_oldness(self):
        if self.__variance_oldness is None:
            self.compute_variance()
        return self.__variance_oldness

class DataRecorderStateVec(Recorder):
    def __init__(self,simulation_count,data_shape=None, distances_matrix=None):
        super().__init__(simulation_count=simulation_count,data_shape=data_shape)
        self.distances_matrix = distances_matrix
        self.__distance = None
        self.__expected_distance = None
        self.__oldness = None
        self.__expected_oldness = None
    
    def fileter_function(self,**kwargs):
        return numpy.array(kwargs["states"])
    

    def __getitem__(self,key):
        if key == "oldness":
            return self.oldness
        elif key == "distance":
            return self.distance
        elif key == "expected_distance":
            return self.expected_distance
        elif key == "expected_oldness":
            return self.expected_oldness
        elif key == "record":
            return self.record
        else:
            raise ValueError(f"Unknown key: {key}")
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
    def keys(self):
        return ["record","oldness","distance","expected_distance","expected_oldness","record"]
    def compute_oldness(self):#todo
        agents_count = len(self._Recorder__return_record.shape) - 1
        if self.__oldness is None:
            axis_list = [tuple([j  for j in range(agents_count) if i != j])for i in range(agents_count)]
            variants_count = self._Recorder__return_record.shape[1]
            self.__oldness = numpy.array([[t*numpy.sum(numpy.sum(self._Recorder__return_record[t], axis=axis_list[ai])*numpy.arange(variants_count)) for ai in range(agents_count)] for t in range(self._Recorder__simulation_count)])
            self.__expected_oldness = numpy.sum(self.__oldness[self._Recorder__simulation_count//10:], axis=0)
    @property
    def simulation_count(self):
        return self._Recorder__simulation_count
    @property
    def oldness(self):
        if self.__oldness is None:
            self.compute_oldness()
        return self.__oldness
    
    @property
    def expected_distance(self):
        if self.__expected_distance is None:
            self.compute_distance()
        return self.__expected_distance
    
    @property
    def expected_oldness(self):
        if self.__expected_oldness is None:
            self.compute_oldness()
        return self.__expected_oldness
    @property
    def distance(self):
        if self.__distance is None:
            self.compute_distance()
        return self.__distance
        # return numpy.tensordot( self.distances_matrix,self._Recorder__return_record, axes=(range(2, agents_count + 2), range(1, agents_count + 1)))

        # distances_record =numpy.tensordot(states_record, distances_matrix, axes=(range(1, agents_count + 1), range(2, agents_count + 2)))

    def compute_distance_by_origin(self):
        """
        各origin（各エージェント起点）ごとに、全エージェント間のdistance（origin由来単語集合の対称差の大きさ）を計算して返す。
        shape: (agents_count, agents_count, agents_count)
        [origin, i, j] = originごとにi, j間のdistance
        """
        record = self._Recorder__return_record  # (simulation_count, ...)
        agents_count = record.shape[1]
        distance_by_origin = numpy.zeros((agents_count, agents_count, agents_count))
        for origin in range(agents_count):
            for i in range(agents_count):
                nonzero_indices_i = numpy.argwhere(record[:, i, ...] > 0)
                words_i = set(map(tuple, nonzero_indices_i))
                words_i_origin = set([w for w in words_i if w[1] == origin])
                for j in range(agents_count):
                    nonzero_indices_j = numpy.argwhere(record[:, j, ...] > 0)
                    words_j = set(map(tuple, nonzero_indices_j))
                    words_j_origin = set([w for w in words_j if w[1] == origin])
                    distance_by_origin[origin, i, j] = len(words_i_origin.symmetric_difference(words_j_origin))
        self.__distance_by_origin = distance_by_origin

    @property
    def distance_by_origin(self):
        if not hasattr(self, "__distance_by_origin") or self.__distance_by_origin is None:
            self.compute_distance_by_origin()
        return self.__distance_by_origin
class DistanceRecorder(Recorder):
    def __init__(self,simulation_count,data_shape=None):
        super().__init__(simulation_count=simulation_count,data_shape=data_shape)
    
    def fileter_function(self,**kwargs):
        return numpy.array([[numpy.abs(agenti.data-agentj.data).sum() for agenti in kwargs["agents"]] for agentj in kwargs["agents"]])

class FilterFunction:#argsは(data,agents)
    def filter_data(**kwargs):
        return kwargs["a"]
    

