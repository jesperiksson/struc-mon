from dataclasses import dataclass, field
import scipy.stats as stats

class Classifier():
    def __init__(self,base_population):
        self.base_population = base_population
        
    def classify(self,prediction):
        return stats.ttest_ind(
            a = prediction,
            b = self.base_population,
            axis = 0,
            equal_var = False,
        )
        
