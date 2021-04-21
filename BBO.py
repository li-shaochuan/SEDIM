import numpy as np
import random as rm
from generateArtich import Generator,randomSpaceSelector
rm.seed(1)



class BBO():
    def __init__(self,Elite_size,PopSize,key,pmutate):
        self.Elite_size=Elite_size
        self.mu=list()
        self.lambda1=list()
        self.PopSize=PopSize
        self.keys=key
        self.pmutate=pmutate
        for i in range(PopSize):
            self.mu.append((PopSize + 1 - (i)) / (PopSize + 1))
            self.lambda1.append( 1 - self.mu[i])

    def remove_Dup(self,pop):
        net_g = Generator()
        PopSize = len(pop)
        Dup_ind_index=list()
        for i in range(PopSize):
            ind1 = pop[i]
            for j in range(i + 1, PopSize):
                ind2 = pop[j]
                sam_num = 0
                for key in ind1.keys():
                    if ind1[key] == ind2[key]:
                        sam_num += 1
                if sam_num == len(ind1.keys()):
                    print('Reset the same net:')
                    print(ind1)
                    print(ind2)
                    pop[j] = net_g.randomNetInit()
                    Dup_ind_index.append(j)
        return pop,Dup_ind_index
    def sort(self,population,fitness):
        sort_index = np.argsort(np.array(fitness))
        _ = list()
        fit = list()
        for i in range(len(sort_index)):
            _.append(population[sort_index[i]])
            fit.append(fitness[sort_index[i]])
        return _,fit

    def inspect_net_parameters(self,pop):
        for net in pop:
            nb_layers = net.get('nb_layers')
            selector = randomSpaceSelector()
            nb_neurons=net.get('nb_neurons')

            net['nb_neurons'] =nb_neurons
            # Mutate one of the params.
            for key in net.keys():
                if isinstance(net.get(key), list):
                    _ = selector.get_sapce(key, 'regression')
                    for i in range(nb_layers - len(net.get(key))):
                        net.get(key).append(rm.choice(_))
            for i in range(nb_layers):
                if nb_neurons[i]==0:
                    nb_neurons[i]=selector.random_pick_nb_neurons()
            net['nb_neurons'] = sorted(net['nb_neurons'], reverse=True)
        return pop


    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        # Choose a random key.
        _=None

        mutation=None
        while _ ==None:
            mutation = rm.choice(list(network.keys()))
            selector=randomSpaceSelector()

            # Mutate one of the params.
            _=selector.get_sapce(mutation, 'regression')
        print('mute:' + mutation)
        if isinstance(network[mutation],list) is not True:
            network[mutation] = rm.choice(_)
        else:
            if mutation=='nb_neurons':
                network[mutation] = sorted([rm.choice(_) for i in range(network.get('nb_layers'))],reverse=True)
            else:
                network[mutation] =[rm.choice(_) for i in range(network.get('nb_layers'))]
        return network
    def evolue(self,population,fitness):
        population,fitness=self.sort(population,fitness)
        EliteSolution=list()
        EliteCost=list()
        Island=list()
        for i in range(self.PopSize):
            Island.append(population[i])
        for i in range(self.Elite_size):
            EliteSolution.append(population[i])
            EliteCost.append(fitness[i])
        # Performing Migration operator
        for k in range(self.PopSize):
            for j in self.keys:
                if rm.random() < self.lambda1[k]:
                    # Performing Roulette Wheel
                    RandomNum = rm.random() * np.sum(self.mu)
                    Select = self.mu[1]
                    SelectIndex = 0
                    while (RandomNum > Select) and (SelectIndex < (self.PopSize - 1)):
                        SelectIndex = SelectIndex + 1
                        Select = Select + self.mu[SelectIndex]

                    Island[k][j] = population[SelectIndex][j]
                else:
                    Island[k][j] = population[k][j]
        # Performing Mutation
        for k in range(self.PopSize):
            if self.pmutate > rm.random():
                Island[k]=self.mutate(population[k])
        # Performing the bound checking
        Island=self.inspect_net_parameters(Island)
        #return the habitats with their new versions.
        return Island,EliteSolution,EliteCost

