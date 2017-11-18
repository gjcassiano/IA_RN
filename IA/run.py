
import sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
from ga_numeric import genetic_algorithm
from matyas import Matyas


NumIndividuals = 80
MinX1 = -10
MaxX1 = 10
MinX2 = -10
MaxX2 = 10
IndividualSize = 16
MutationRate = 0.02

problem = Matyas(MinX1, MaxX1, MinX2, MaxX2, IndividualSize)

MaxGeneration = 100

Target = 0.00001
Elitism = True

ClassHandle  = genetic_algorithm(problem,MutationRate,Elitism) 
fit,generation = ClassHandle.search(NumIndividuals, MaxGeneration, Target)

interaction=[i for i in range(generation)]
plt.plot(interaction,fit)
plt.show()  

