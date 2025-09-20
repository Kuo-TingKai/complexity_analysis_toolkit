"""
Information Theory Analysis Module

This module implements information-theoretic measures for system complexity:
- Synergy: measures emergent information in the whole system
- Phi (Φ): integrated information measure from IIT theory
- Multi-information: total correlation among system components
"""

import numpy as np
from scipy.stats import entropy
from scipy.optimize import minimize
import itertools
from typing import List, Tuple, Dict, Optional
import warnings


class InformationTheoryAnalyzer:
    """
    Analyzer for information-theoretic complexity measures
    """
    
    def __init__(self, binning_method: str = 'uniform', n_bins: int = 10):
        """
        Initialize the information theory analyzer
        
        Args:
            binning_method: Method for discretizing continuous data ('uniform', 'quantile')
            n_bins: Number of bins for discretization
        """
        self.binning_method = binning_method
        self.n_bins = n_bins
        
    def _discretize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Discretize continuous data into bins
        
        Args:
            data: Input data array
            
        Returns:
            Discretized data
        """
        if self.binning_method == 'uniform':
            min_val, max_val = np.min(data), np.max(data)
            bins = np.linspace(min_val, max_val, self.n_bins + 1)
            return np.digitize(data, bins) - 1
        elif self.binning_method == 'quantile':
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bins = np.percentile(data, percentiles)
            return np.digitize(data, bins) - 1
        else:
            raise ValueError(f"Unknown binning method: {self.binning_method}")
    
    def _joint_entropy(self, *variables: np.ndarray) -> float:
        """
        Calculate joint entropy of multiple variables
        
        Args:
            *variables: Variable arrays
            
        Returns:
            Joint entropy value
        """
        if len(variables) == 1:
            return entropy(np.bincount(variables[0]) / len(variables[0]), base=2)
        
        # Create joint histogram
        joint_hist, _ = np.histogramdd(np.column_stack(variables), bins=self.n_bins)
        joint_probs = joint_hist / np.sum(joint_hist)
        joint_probs = joint_probs[joint_probs > 0]  # Remove zero probabilities
        
        return entropy(joint_probs, base=2)
    
    def synergy(self, data: np.ndarray) -> float:
        """
        Calculate synergy - emergent information in the whole system
        
        Synergy measures information that emerges only when considering
        the system as a whole, not present in any subset of components.
        
        Args:
            data: System data where columns are variables/components
            
        Returns:
            Synergy value
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_vars = data.shape[1]
        if n_vars < 2:
            return 0.0
            
        # Discretize data
        discrete_data = np.array([self._discretize_data(data[:, i]) for i in range(n_vars)]).T
        
        # Calculate total multi-information
        total_multi_info = self.multi_information(discrete_data)
        
        # Calculate maximum possible synergy (when all variables are independent)
        individual_entropies = [self._joint_entropy(discrete_data[:, i]) for i in range(n_vars)]
        joint_entropy = self._joint_entropy(*[discrete_data[:, i] for i in range(n_vars)])
        
        # Synergy is the difference between joint entropy and sum of individual entropies
        synergy = sum(individual_entropies) - joint_entropy
        
        return max(0, synergy)  # Ensure non-negative
    
    def phi(self, data: np.ndarray, partition: Optional[List[List[int]]] = None) -> float:
        """
        Calculate Phi (Φ) - integrated information measure
        
        Phi measures the information integrated by a system that is lost
        when it is partitioned into independent parts.
        
        Args:
            data: System data where columns are variables/components
            partition: Optional partition of variables (if None, uses minimal information partition)
            
        Returns:
            Phi value
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_vars = data.shape[1]
        if n_vars < 2:
            return 0.0
            
        # Discretize data
        discrete_data = np.array([self._discretize_data(data[:, i]) for i in range(n_vars)]).T
        
        if partition is None:
            # Find minimal information partition (MIP)
            partition = self._find_minimal_information_partition(discrete_data)
        
        # Calculate effective information for the partition
        ei = self._effective_information(discrete_data, partition)
        
        return ei
    
    def _find_minimal_information_partition(self, data: np.ndarray) -> List[List[int]]:
        """
        Find the minimal information partition (MIP) that minimizes effective information
        
        Args:
            data: Discretized system data
            
        Returns:
            Partition that minimizes effective information
        """
        n_vars = data.shape[1]
        
        # Generate all possible bipartitions
        best_partition = None
        min_ei = float('inf')
        
        for i in range(1, 2**n_vars // 2):
            partition = [[], []]
            for j in range(n_vars):
                partition[(i >> j) & 1].append(j)
            
            if len(partition[0]) > 0 and len(partition[1]) > 0:
                ei = self._effective_information(data, partition)
                if ei < min_ei:
                    min_ei = ei
                    best_partition = partition
        
        return best_partition if best_partition else [[i for i in range(n_vars)]]
    
    def _effective_information(self, data: np.ndarray, partition: List[List[int]]) -> float:
        """
        Calculate effective information for a given partition
        
        Args:
            data: Discretized system data
            partition: Partition of variables
            
        Returns:
            Effective information value
        """
        if len(partition) != 2:
            raise ValueError("Partition must have exactly 2 parts")
        
        # Calculate entropies for each part of the partition
        part1_data = [data[:, i] for i in partition[0]]
        part2_data = [data[:, i] for i in partition[1]]
        
        h_part1 = self._joint_entropy(*part1_data)
        h_part2 = self._joint_entropy(*part2_data)
        
        # Calculate joint entropy of the entire system
        all_data = [data[:, i] for i in range(data.shape[1])]
        h_joint = self._joint_entropy(*all_data)
        
        # Effective information is the difference
        ei = h_part1 + h_part2 - h_joint
        
        return max(0, ei)  # Ensure non-negative
    
    def multi_information(self, data: np.ndarray) -> float:
        """
        Calculate multi-information (total correlation)
        
        Multi-information measures the total amount of information
        shared among all variables in the system.
        
        Args:
            data: System data where columns are variables/components
            
        Returns:
            Multi-information value
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_vars = data.shape[1]
        if n_vars < 2:
            return 0.0
            
        # Discretize data
        discrete_data = np.array([self._discretize_data(data[:, i]) for i in range(n_vars)]).T
        
        # Calculate individual entropies
        individual_entropies = [self._joint_entropy(discrete_data[:, i]) for i in range(n_vars)]
        
        # Calculate joint entropy
        joint_entropy = self._joint_entropy(*[discrete_data[:, i] for i in range(n_vars)])
        
        # Multi-information is the difference
        multi_info = sum(individual_entropies) - joint_entropy
        
        return max(0, multi_info)  # Ensure non-negative
    
    def analyze(self, data: np.ndarray) -> Dict[str, float]:
        """
        Perform comprehensive information-theoretic analysis
        
        Args:
            data: System data where columns are variables/components
            
        Returns:
            Dictionary containing all information-theoretic measures
        """
        results = {}
        
        try:
            results['synergy'] = self.synergy(data)
        except Exception as e:
            warnings.warn(f"Failed to calculate synergy: {e}")
            results['synergy'] = np.nan
            
        try:
            results['phi'] = self.phi(data)
        except Exception as e:
            warnings.warn(f"Failed to calculate phi: {e}")
            results['phi'] = np.nan
            
        try:
            results['multi_information'] = self.multi_information(data)
        except Exception as e:
            warnings.warn(f"Failed to calculate multi-information: {e}")
            results['multi_information'] = np.nan
        
        return results
