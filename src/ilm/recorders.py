import numpy


def recorder(recorder_type=None, simulation_count=None, agents_count=None):
    if recorder_type is None:
        return Recorder(simulation_count=simulation_count)
    elif recorder_type == "data":
        return DataRecorder(simulation_count=simulation_count)


class Recorder:
    def __init__(self,simulation_count,data_shape=None):
        self.__simulation_count = simulation_count
        if not simulation_count is None and not data_shape is None:
            self.__return_record = numpy.empty(((simulation_count,)+ data_shape))
        else:
            self.__return_record = None
        if recorder_type == "data":
            self.__filter_function = FilterFunction.filter_data
        self.__i=0
        
    def __call__(self,**kwargs):
        tmp_record=self.fileter_function(**kwargs)
        if self.__return_record is None:
            self.__return_record=numpy.empty((self.__simulation_count)+tmp_record.shape)
        self.__return_record[self.__i]=tmp_record
        self.__i+=1
    
    #そのまま返すとreturn_recordの初期化でエラー
    # def fileter_function(self,**kwargs):
    #     return **kwargs

    @property
    def record(self):
        return self.__return_record

class DataRecorder(Recorder):
    def __init__(self,simulation_count,data_shape=None):
        super().__init__(simulation_count=simulation_count,data_shape=data_shape)
    
    def fileter_function(self,**kwargs):
        return kwargs[data]


class FilterFunction:#argsは(data,agents)
    def filter_data(**kwargs):
        return kwargs["data"]
        
