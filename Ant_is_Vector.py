
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random
import math
import time
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor


class AntIS():
    def get_pairwise_distance(self, matrix: np.ndarray) -> np.ndarray:
        return euclidean_distances(matrix)

    def get_visibility_rates_by_distances(self, distances: np.ndarray) -> np.ndarray:
        distances = np.where(distances == 0, 1e-9, distances)
        visibilities = 1 / distances
        np.fill_diagonal(visibilities, 0)
        return visibilities

    def create_colony(self, num_ants):
        return np.full((num_ants, num_ants), -1)

    def create_pheromone_trails(self, num_cities: int, initial_pheromone: float) -> np.ndarray:
        trails = np.ones((num_cities, num_cities)) * initial_pheromone
        np.fill_diagonal(trails, 0)
        return trails

    def get_pheromone_deposit(self, ant_choices: List[Tuple[int, int]], distances: np.ndarray, deposit_factor: float) -> float:
        tour_length = np.sum(distances[path[0], path[1]] for path in ant_choices)
        if tour_length == 0:
            return 0
        if math.isinf(tour_length):
            print('ERROR! Length of the tour is infinity.')
        return deposit_factor / tour_length

    def get_probabilities_paths_ordered(self, ant: np.ndarray, visib_rates: np.ndarray, phe_trails: np.ndarray) -> Tuple[Tuple[int, float]]:
        available_instances = np.nonzero(ant < 0)[0]
        smell = np.sum(phe_trails[available_instances] * visib_rates[available_instances])
        path_smell = phe_trails[available_instances] * visib_rates[available_instances]
        probabilities = np.zeros((len(available_instances), 2))
        probabilities[:, 0] = available_instances
        probabilities[:, 1] = np.divide(path_smell, smell, out=np.zeros_like(path_smell), where=path_smell != 0)
        sorted_probabilities = probabilities[probabilities[:, 1].argsort()][::-1]
        return tuple([(int(i[0]), i[1]) for i in sorted_probabilities])

    def evaluate_solution(self, instances_selected: np.ndarray, X, Y) -> float:
        X_train = X[instances_selected, :]
        Y_train = Y[instances_selected]
        classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)
        Y_pred = classifier_1nn.predict(X)
        return accuracy_score(Y, Y_pred)

    def ant_search_solution(self, i, ant, Q, evaporation_rate, ant_choices, distances, visibility_rates, pheromone_trails):
        while -1 in ant:
            last_choice = ant_choices[i][-1]
            ant_pos = last_choice[1]
            choices = self.get_probabilities_paths_ordered(
                ant,
                visibility_rates[ant_pos, :],
                pheromone_trails[ant_pos, :])

            for choice in choices:
                next_instance = choice[0]
                probability = choice[1]
                ajk = random.randint(0, 1)
                final_probability = probability * ajk
                if final_probability != 0:
                    ant_choices[i].append((ant_pos, next_instance))
                    ant[next_instance] = 1
                    pheromone_trails[ant_pos, next_instance] += self.get_pheromone_deposit(ant_choices[i], distances, Q)
                else:
                    ant[next_instance] = 0

    def run_colony(self, X, Y, initial_pheromone, evaporation_rate, Q):
        distances = self.get_pairwise_distance(X)
        visibility_rates = self.get_visibility_rates_by_distances(distances)
        the_colony = self.create_colony(X.shape[0])
        evaporation_rate = evaporation_rate
        for i in range(X.shape[0]):
            the_colony[i, i] = 1
        ant_choices = [[(i, i)] for i in range(len(the_colony))]
        pheromone_trails = self.create_pheromone_trails(X.shape[0], initial_pheromone)
        with ThreadPoolExecutor(max_workers=1) as executor:
            for i, ant in enumerate(the_colony):
                executor.submit(self.ant_search_solution, i, ant, Q, evaporation_rate, ant_choices, distances, visibility_rates, pheromone_trails)
        best_solution = self.get_best_solution(the_colony, X, Y)
        instances_selected = np.nonzero(best_solution)[0]
        return instances_selected

    def get_best_solution(self, ant_solutions: np.ndarray, X, Y) -> np.ndarray:
        accuracies = [self.evaluate_solution(np.nonzero(solution)[0], X, Y) for solution in ant_solutions]
        best_solution_index = np.argmax(accuracies)
        best_solution = ant_solutions[best_solution_index]
        return best_solution


if __name__ == '__main__':
    start_time = time.time()
    original_df = pd.read_csv("DEMANDA SIC_sea_dados_inputados.csv", sep=',') 
    dataframe = pd.read_csv("DEMANDA SIC_sea_dados_inputados.csv", sep=',')
    classes = dataframe["DW_STATUS_NOT_REATIVADO"]
    dataframe = dataframe.drop(columns=["DW_STATUS_NOT_REATIVADO"])
    initial_pheromone = 1
    Q = 1
    evaporation_rate = 0.1
    ants = AntIS()
    print('Starting search')
    indices_selected = ants.run_colony(dataframe.to_numpy(), classes.to_numpy(), initial_pheromone, evaporation_rate, Q)
    print('End Search')
    print(len(indices_selected))
    reduced_dataframe = original_df.iloc[indices_selected]
    reduced_dataframe.to_csv('Demanda_Reduzido_vector10sv', index=False)
    print("Execution finished")
    print("--- %s Hours ---" % ((time.time() - start_time) // 3600))
    print("--- %s Minutes ---" % ((time.time() - start_time) // 60))
    print("--- %s Seconds ---" % (time.time() - start_time))
