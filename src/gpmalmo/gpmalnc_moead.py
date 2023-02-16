import numpy as np

from gpmalmo import rundata as rd
from gptools.moead import MOEAD


class GPMALNCMOEAD(MOEAD):

    def __init__(self, data_t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_t = data_t
        self.functionType_ = "gpMalNC"
        if rd.nobj==2:
            self.obj_mins = [0, 1]
            self.obj_maxes = [1, int(1e6)]
        elif rd.nobj==3:
            self.obj_mins = [0, 0, 1]
            self.obj_maxes = [1, int(1e6), rd.num_trees]
        else:
            raise ValueError(self.data)
            print(self.data)
        self.DECOMPOSITION = 'tchebycheff'

    def updateProblem(self, individual, id_, type_):
        """
        individual : A new candidate individual
        id : index of the subproblem
        type : update solutions in neighbourhood (type = 1) or whole population otherwise.
        """
        size = int()
        time = int()
        time = 0

        if type_ == 1:
            size = len(self.neighbourhood_[id_])
        else:
            size = len(self.population)
        perm = [None] * size

        self.randomPermutations(perm, size)

        for i in range(size):
            k = int()
            if type_ == 1:
                k = self.neighbourhood_[id_][perm[i]]
            else:
                k = perm[i]

            f1 = self.skew_fitness(self.population[k].fitness.values, self.lambda_[k])
            f2 = self.skew_fitness(individual.fitness.values, self.lambda_[k])

            if f2 < f1:  # minimization, JMetal default
                # if f2 >= f1:  # maximization assuming DEAP weights paired with fitness
                self.population[k] = individual
                time += 1
            if time >= self.nr_:
                self.paretoFront.update(self.population)
                return

    def skew_fitness(self, fitness, lambda_):
        fitness = np.array(fitness)
        fitness -= self.obj_mins
        fitness /= self.obj_maxes
        fitness *= lambda_

        if self.DECOMPOSITION == 'tchebycheff':
            return fitness.max()
        elif self.DECOMPOSITION == 'weighted':
            return fitness.sum()
        else:
            raise NotImplementedError

    def fitnessFunction(self, individual, lambda_):
        raise ValueError
        #fitness = self.fitness_function(self.data_t, self.toolbox, individual)
        #return self.skew_fitness(fitness.values, lambda_)

        #return self.skew_fitness(individual.fitness.values, lambda_)
