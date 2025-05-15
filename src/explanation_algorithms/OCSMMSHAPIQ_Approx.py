from itertools import product
import numpy as np
from scipy.special import bernoulli, binom
from shapiq.interaction_values import InteractionValues, finalize_computed_interactions
from torch import FloatTensor
import torch
import copy
from typing import List
from dataclasses import field
from tqdm import tqdm
from joblib import Parallel, delayed
from shapiq.utils.sets import powerset
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.plot.network import network_plot

def display_network_plot(interactions: InteractionValues, feature_names):
    network_plot(interaction_values=interactions, feature_names=feature_names, draw_legend=False, show=True)


def solve_regression(X: np.ndarray, y: np.ndarray, kernel_weights: np.ndarray) -> np.ndarray:
    try:
        # try solving via solve function
        WX = kernel_weights[:, np.newaxis] * X
        phi = np.linalg.solve(X.T @ WX, WX.T @ y)
    except np.linalg.LinAlgError:
        # solve WLSQ via lstsq function and throw warning
        W_sqrt = np.sqrt(kernel_weights)
        X = W_sqrt[:, np.newaxis] * X
        y = W_sqrt * y
        phi = np.linalg.lstsq(X, y, rcond=None)[0]
    return phi


class OCSMMSHAPIQ:
    def __init__(self, n: int, max_order: int, random_state: int | None = None):
        self.n = n
        self.max_order = max_order
        self._big_M = 1_000_000_000
        self.num_cpus = 120
        self._bernoulli_numbers = bernoulli(n)
        self._grand_coalition_set = set(range(n))
        self.cme_regularisation: FloatTensor = field(init=False, default=torch.tensor(1e-4).float())
        self.interaction_lookup = self._build_interaction_lookup()
            
        sampling_weights = self._init_sampling_weights()
        self._sampler = CoalitionSampler(
            n_players=self.n,
            sampling_weights=sampling_weights,
            pairing_trick=False,
            random_state=random_state,
        )


    def _init_sampling_weights(self) -> np.ndarray:
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(0, self.n + 1):
            if (coalition_size < self.max_order) or (coalition_size > self.n - self.max_order):
                weight_vector[coalition_size] = self._big_M
            else:
                weight_vector[coalition_size] = 1 / (coalition_size * (self.n - coalition_size))
        sampling_weight = weight_vector / np.sum(weight_vector)
        return sampling_weight


    def _build_interaction_lookup(self):  #mapping between interaction subsets and indices.
        lookup = {}
        idx = 0
        for order in range(0, self.max_order + 1):
            for interaction in powerset(self._grand_coalition_set, min_size=order, max_size=order):
                lookup[tuple(sorted(interaction))] = idx
                idx += 1
        return lookup

    def _init_kernel_weights(self, interaction_size: int) -> np.ndarray: #  W
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(0, self.n + 1):
            if (coalition_size < interaction_size) or (
                coalition_size > self.n - interaction_size
            ):
                weight_vector[coalition_size] = self._big_M
            else:
                weight_vector[coalition_size] = 1 / (
                    (self.n - 2 * interaction_size + 1)
                    * binom(self.n - 2 * interaction_size, coalition_size - interaction_size)
                )
        return weight_vector

    def _bernoulli_weights(self, intersection_size: int, interaction_size: int) -> float:   # lambda
        weight = 0
        for sum_index in range(1, intersection_size + 1):
            weight += (
                binom(intersection_size, sum_index)
                * self._bernoulli_numbers[interaction_size - sum_index]
            )
        return weight

    def _get_bernoulli_weights_matrix(self) -> np.ndarray:   # lambda matrix
        bernoulli_weights = np.zeros((self.max_order + 1, self.max_order + 1))
        for interaction_size in range(1, self.max_order + 1):
            for intersection_size in range(interaction_size + 1):
                bernoulli_weights[interaction_size, intersection_size] = self._bernoulli_weights(
                    intersection_size,
                    interaction_size,
                )
        return bernoulli_weights

    def _get_regression_matrices(
        self,
        max_order: int,
        kernel_weights: np.ndarray,
        regression_coeff_weights: np.ndarray,
        sampling_adjustment_weights: np.ndarray,
        coalitions_matrix: np.ndarray
    ):
        interaction_masks = []
        interaction_sizes = []

        for interaction_size in range(0, max_order + 1):
            for interaction in powerset(range(self.n), min_size=interaction_size, max_size=interaction_size):
                mask = np.zeros(self.n, dtype=int)
                mask[list(interaction)] = 1
                interaction_masks.append(mask)
                interaction_sizes.append(interaction_size)

        interaction_masks = np.array(interaction_masks).T  # Shape: (n, n_interactions)
        interaction_sizes = np.array(interaction_sizes)  # Shape: (n_interactions,)

        intersection_sizes = coalitions_matrix @ interaction_masks
        regression_matrix = regression_coeff_weights[interaction_sizes, intersection_sizes]
        regression_weights = kernel_weights[np.sum(coalitions_matrix, axis=1)]
        regression_weights *= sampling_adjustment_weights

        return regression_matrix, regression_weights

    
    def _find_interactions_single_group(self, game_values: np.ndarray, coalitions_matrix) -> dict:
        regression_coeff_weights = self._get_bernoulli_weights_matrix()   # lambda matrix
        kernel_weights_dict = {
            i: self._init_kernel_weights(i) for i in range(1, self.max_order + 1)   # W
        }

        sampling_adjustment_weights = self._sampler.sampling_adjustment_weights
        coalitions_size = np.sum(coalitions_matrix, axis=1)        
        empty_coalition_value = float(game_values[coalitions_size == 0][0])
        residual_game_values = {1: copy.copy(game_values)}
        residual_game_values[1] -= empty_coalition_value
        sii_values = np.array([empty_coalition_value])

        idx_order = 1 
        for interaction_size in range(1, self.max_order + 1):
            regression_matrix, regression_weights = self._get_regression_matrices(
                kernel_weights=kernel_weights_dict[interaction_size],
                regression_coeff_weights=regression_coeff_weights,
                sampling_adjustment_weights=sampling_adjustment_weights,
                coalitions_matrix=coalitions_matrix,
                max_order=self.max_order
            )
            n_interactions = int(binom(self.n, interaction_size))
            regression_matrix = regression_matrix[:, idx_order : idx_order + n_interactions]
            idx_order += n_interactions

            sii_values_current_size = solve_regression(
                    X=regression_matrix,
                    y=residual_game_values[interaction_size],
                    kernel_weights=regression_weights,
                )
            approximations = np.dot(regression_matrix, sii_values_current_size)
            sii_values = np.hstack((sii_values, sii_values_current_size))
            residual_game_values[interaction_size + 1] = (
                residual_game_values[interaction_size] - approximations
            )

        baseline_value = float(game_values[self._sampler.empty_coalition_index])
        # print(sii_values)

        interactions = InteractionValues(
            values=sii_values,
            index="SII",
            interaction_lookup=self.interaction_lookup,
            baseline_value=baseline_value,
            min_order=0,
            max_order=self.max_order,
            n_players=self.n,
            estimated=False if self.budget >= 2**self.n else True,
            estimation_budget=self.budget,
        )

        return finalize_computed_interactions(interactions, target_index="SII")


    def explain_single_group(self, group_data: torch.Tensor, X: List[torch.Tensor], model, regularization: float, budget: int) -> dict:

        self._sampler.sample(budget)
        coalitions_matrix = self._sampler.coalitions_matrix
        self.budget = budget

        
        def compute_value_function(S):
            S_tensor = torch.tensor(S).bool()

            group_S = group_data[:, S_tensor]
            X_train_S = [g[:, S_tensor] for g in X]

            K_SS = torch.tensor(model.kappa_matrix(X_train_S, X_train_S))
            K_xSS = torch.tensor(model.kappa_matrix([group_S], X_train_S)) # shape (m_support,)

            regularization_term = len(X) * regularization
            K_SS_regularized = K_SS + regularization_term * torch.eye(K_SS.shape[0]).float()
            K_SS_inv = torch.inverse(K_SS_regularized)
            conditional_mean_projection = K_xSS @ K_SS_inv
            decision = torch.tensor(model.decision+model.rho)
            value = conditional_mean_projection @ decision
            return value.item()


        game_values = np.array(
            Parallel(n_jobs=self.num_cpus)(
                delayed(compute_value_function)(S) for S in tqdm(coalitions_matrix)
            )
        )
        return self._find_interactions_single_group(game_values, coalitions_matrix)
