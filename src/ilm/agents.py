import numpy 


class BayesianFiniteVariantsAgent:
    def __init__(self,data=None , alpha:float=0.0 ):
        self.__alpha=alpha
        if not data is None:
            self.__data_size=numpy.sum(data)
            self.__hypothesis=numpy.empty(self.__data_size)
            self.learn(data)
        else:
            if data_size is None:
                raise Exception
            self.__data_size=data_size
            self.__hypothesis=numpy.zeros(self.__data_size)
        
    def learn(self , data:numpy.array):
        tmp=numpy.array([(x+self.__alpha/self.variants_count) for x in data])
        self.__hypothesis=tmp/numpy.sum(tmp)
    def produce(self , n=None):
        if n is None:
            n=self.variants_count
        return numpy.random.multinomial(n=n,pvals=self.__hypothesis)
    
    @property
    def hypothesis(self) -> numpy.array:
        return self.__hypothesis
    
    @property
    def data_size(self) -> int:
        return self.__data_size

class BayesianInfiniteVariantsAgent:
    def __init__(self,agentgent_number=None,generation=None,data=None, alpha:float=0.0):
        if agentgent_number==None:
            self.__agent_number=0
        else:
            self.__agent_number=agentgent_number
        if generation is None:
            self.__generation=0
        else:
            self.__generation=generation
        self.__alpha=alpha
        if data is None:
            self.__data=data
        elif data.ndim == 1:#data[generation, agent, number]
            self.__data=numpy.array([[self.__generation, self.__agent_number, d] for d in data]).astype(numpy.uint8,casting='unsafe')
        else:
            self.__data=data
        self.__data_size:numpy.uint8=len(data)
        self.__alpha=alpha
    def learn(self , data):
        self.__data=data
        self.__data_size=data.shape[0]
        self.__generation+=1

    def produce(self, n=None):
        ret_data=numpy.empty(n)
        if n is None:
            n=self.__data_size
        new_word_count=numpy.random.binomial(n,self.__alpha/(self.__data_size+self.__alpha))
        if new_word_count==0:
            return self.__data[numpy.random.randint(self.__data_size, size=n-new_word_count), :]
        else:
            return numpy.concatenate([self.__data[numpy.random.randint(self.__data_size, size=n-new_word_count), :], numpy.array([[self.__generation,self.__agent_number,i] for i in range(new_word_count)])],dtype=numpy.uint8,casting="unsafe")

    @property
    def data_size(self) -> int:
        return self.__data_size


