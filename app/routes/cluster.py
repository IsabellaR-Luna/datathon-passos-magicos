
#Para embeddings model
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

#Bibliotecas para análise de Clusters
import umap
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
from functools import partial
import hdbscan
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import StandardScaler

class Agrupamento():

    def __init__(self, dados):
        self.df = dados
    
    #com busca de hiper para Umap:
    def generate_clusters(message_embeddings,
                        n_neighbors,
                        n_components,
                        min_cluster_size,
                        min_samples = None,
                        random_state = None):
        """
        Returns HDBSCAN objects after first performing dimensionality reduction using UMAP

        Arguments:
            message_embeddings: embeddings to use
            n_neighbors: int, UMAP hyperparameter n_neighbors
            n_components: int, UMAP hyperparameter n_components
            min_cluster_size: int, HDBSCAN hyperparameter min_cluster_size
            min_samples: int, HDBSCAN hyperparameter min_samples
            random_state: int, random seed

        Returns:
            clusters: HDBSCAN object of clusters
        """

        umap_embeddings = (umap.UMAP(n_neighbors = n_neighbors,
                                    n_components = n_components,
                                    metric = 'cosine',
                                    random_state=random_state)
                                .fit_transform(message_embeddings))

        clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                                min_samples = min_samples,
                                metric='euclidean',
                                gen_min_span_tree=True,
                                cluster_selection_method='eom').fit(umap_embeddings)

        return clusters

    def score_clusters(clusters, prob_threshold = 0.05):
        """
        Returns the label count and cost of a given clustering

        Arguments:
            clusters: HDBSCAN clustering object
            prob_threshold: float, probability threshold to use for deciding
                            what cluster labels are considered low confidence

        Returns:
            label_count: int, number of unique cluster labels, including noise
            cost: float, fraction of data points whose cluster assignment has
                a probability below cutoff threshold
        """

        cluster_labels = clusters.labels_
        label_count = len(np.unique(cluster_labels))
        total_num = len(clusters.labels_)
        cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)

        return label_count, cost

    def plot_clusters(embeddings, clusters, n_neighbors=15, min_dist=0.1):
        """
        Reduce dimensionality of best clusters and plot in 2D

        Arguments:
            embeddings: embeddings to use
            clusteres: HDBSCAN object of clusters
            n_neighbors: float, UMAP hyperparameter n_neighbors
            min_dist: float, UMAP hyperparameter min_dist for effective
                    minimum distance between embedded points

        """
        umap_data = umap.UMAP(n_neighbors=n_neighbors,
                            n_components=2,
                            min_dist = min_dist,
                            #metric='cosine',
                            random_state=42).fit_transform(embeddings)

        point_size = 100.0 / np.sqrt(embeddings.shape[0])

        result = pd.DataFrame(umap_data, columns=['x', 'y'])
        result['labels'] = clusters.labels_

        fig, ax = plt.subplots(figsize=(14, 8))
        outliers = result[result.labels == -1]
        clustered = result[result.labels != -1]
        plt.scatter(outliers.x, outliers.y, color = 'lightgrey', s=point_size)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=point_size, cmap='jet')
        plt.colorbar()
        plt.show()

    def bayesian_search(self, embeddings, space, label_lower, label_upper, max_evals=100):
        """
        Perform bayesian search on hyperparameter space using hyperopt

        Arguments:
            embeddings: embeddings to use
            space: dict, contains keys for 'n_neighbors', 'n_components',
                'min_cluster_size', and 'random_state' and
                values that use built-in hyperopt functions to define
                search spaces for each
            label_lower: int, lower end of range of number of expected clusters
            label_upper: int, upper end of range of number of expected clusters
            max_evals: int, maximum number of parameter combinations to try

        Saves the following to instance variables:
            best_params: dict, contains keys for 'n_neighbors', 'n_components',
                'min_cluster_size', 'min_samples', and 'random_state' and
                values associated with lowest cost scenario tested
            best_clusters: HDBSCAN object associated with lowest cost scenario
                        tested
            trials: hyperopt trials object for search

            """

        trials = Trials()
        fmin_objective = partial(self.objective,
                                embeddings=embeddings,
                                label_lower=label_lower,
                                label_upper=label_upper)

        best = fmin(fmin_objective,
                    space = space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)

        best_params = space_eval(space, best)
        print ('best:')
        print (best_params)
        print (f"label count: {trials.best_trial['result']['label_count']}")

        best_clusters = self.generate_clusters(embeddings,
                                        n_neighbors = best_params['n_neighbors'],
                                        n_components = best_params['n_components'],
                                        min_cluster_size = best_params['min_cluster_size'],
                                        min_samples = best_params['min_samples'],
                                        random_state = best_params['random_state'])

        return best_params, best_clusters, trials

    def objective(self, params, embeddings, label_lower, label_upper):
        """
        Objective function for hyperopt to minimize

        Arguments:
            params: dict, contains keys for 'n_neighbors', 'n_components',
                'min_cluster_size', 'random_state' and
                their values to use for evaluation
            embeddings: embeddings to use
            label_lower: int, lower end of range of number of expected clusters
            label_upper: int, upper end of range of number of expected clusters

        Returns:
            loss: cost function result incorporating penalties for falling
                outside desired range for number of clusters
            label_count: int, number of unique cluster labels, including noise
            status: string, hypoeropt status

            """

        clusters = self.generate_clusters(embeddings,
                                    n_neighbors = params['n_neighbors'],
                                    n_components = params['n_components'],
                                    min_cluster_size = params['min_cluster_size'],
                                    random_state = params['random_state'])

        label_count, cost = self.score_clusters(clusters, prob_threshold = 0.05)

        #15% penalty on the cost function if outside the desired range of groups
        if (label_count < label_lower) | (label_count > label_upper):
            penalty = 0.15
        else:
            penalty = 0

        loss = cost + penalty

        return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}
    
    def execute(self):
        features_cluster = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'Defas']
        X = self.df[features_cluster].fillna(self.df[features_cluster].median())
        X_scaled = StandardScaler().fit_transform(X)



        #Definindo o espaço para busca de hiperparâmetros
        hspace = {
            "n_neighbors": hp.choice('n_neighbors', range(3, 30)),
            "n_components": hp.choice('n_components', range(2, 8)),
            "min_cluster_size": hp.choice('min_cluster_size', range(10, 50)),
            "min_samples": hp.choice('min_samples', range(2, 12)),
            "random_state": 42
        }

        label_lower = 3
        label_upper = 5
        max_evals = 25

        best_params_use, best_clusters_use, trials_use = self.bayesian_search(embeddings=X_scaled,
                                                                        space=hspace,
                                                                        label_lower=label_lower,
                                                                        label_upper=label_upper,
                                                                        max_evals=max_evals)

        # Mostrando os melhores hiperparâmetros
        print(best_params_use)

        df["grupo"] = best_clusters_use.labels_

        return df
    
df = pd.DataFrame({
    'IAA': [1, 2, 3, 4, 5],
    'IEG': [5, 4, 3, 2, 1],
    'IPS': [2, 3, 4, 5, 6],
    'IDA': [6, 5, 4, 3, 2],
    'IPV': [3, 4, 5, 6, 7],
    'IAN': [7, 6, 5, 4, 3],
    'Defas': [4, 5, 6, 7, 8]
})

clustering = Agrupamento(df)
result_df = clustering.execute()
print(result_df)