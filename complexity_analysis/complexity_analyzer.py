"""
Comprehensive Complexity Analyzer

This module provides a unified interface for analyzing system complexity using
all three major approaches: information theory, dynamics, and network structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import warnings

from .information_theory import InformationTheoryAnalyzer
from .dynamics import DynamicsAnalyzer
from .network_structure import NetworkStructureAnalyzer


class ComplexityAnalyzer:
    """
    Comprehensive complexity analyzer combining information theory, dynamics, and network structure
    """
    
    def __init__(self, 
                 # Information theory parameters
                 info_binning_method: str = 'uniform',
                 info_n_bins: int = 10,
                 # Dynamics parameters
                 dynamics_tolerance: float = 0.2,
                 dynamics_max_scale: int = 20,
                 # Network structure parameters
                 network_n_clusters: Optional[int] = None,
                 network_clustering_method: str = 'spectral'):
        """
        Initialize the comprehensive complexity analyzer
        
        Args:
            info_binning_method: Binning method for information theory ('uniform', 'quantile')
            info_n_bins: Number of bins for discretization
            dynamics_tolerance: Tolerance for sample entropy calculation
            dynamics_max_scale: Maximum scale for multiscale entropy
            network_n_clusters: Number of clusters for network modularity
            network_clustering_method: Clustering method ('spectral', 'hierarchical')
        """
        self.info_analyzer = InformationTheoryAnalyzer(
            binning_method=info_binning_method,
            n_bins=info_n_bins
        )
        
        self.dynamics_analyzer = DynamicsAnalyzer(
            tolerance=dynamics_tolerance,
            max_scale=dynamics_max_scale
        )
        
        self.network_analyzer = NetworkStructureAnalyzer(
            n_clusters=network_n_clusters,
            clustering_method=network_clustering_method
        )
    
    def analyze_univariate(self, data: np.ndarray, scales: Optional[List[int]] = None) -> Dict[str, Union[float, Dict]]:
        """
        Analyze complexity of univariate time series using dynamics measures
        
        Args:
            data: Univariate time series data
            scales: List of scales for analysis (if None, uses default)
            
        Returns:
            Dictionary containing dynamics-based complexity measures
        """
        if data.ndim > 1:
            if data.shape[1] == 1:
                data = data.flatten()
            else:
                raise ValueError("For univariate analysis, data must be 1D or 2D with single column")
        
        return self.dynamics_analyzer.analyze(data, scales)
    
    def analyze_multivariate(self, data: np.ndarray, scales: Optional[List[int]] = None,
                           network_threshold: Optional[float] = None,
                           network_method: str = 'correlation',
                           hypergraph_k: int = 3) -> Dict[str, Union[float, Dict]]:
        """
        Analyze complexity of multivariate time series using all three approaches
        
        Args:
            data: Multivariate time series data (rows=time, cols=variables)
            scales: List of scales for dynamics analysis
            network_threshold: Threshold for network construction
            network_method: Method for network edge calculation
            hypergraph_k: Size of hyperedges for hypergraph analysis
            
        Returns:
            Dictionary containing all complexity measures
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        results = {}
        
        # Information theory analysis
        try:
            info_results = self.info_analyzer.analyze(data)
            results['information_theory'] = info_results
        except Exception as e:
            warnings.warn(f"Information theory analysis failed: {e}")
            results['information_theory'] = {
                'synergy': np.nan,
                'phi': np.nan,
                'multi_information': np.nan
            }
        
        # Dynamics analysis (for each variable individually)
        try:
            dynamics_results = {}
            for i in range(data.shape[1]):
                var_name = f'variable_{i}'
                dynamics_results[var_name] = self.dynamics_analyzer.analyze(data[:, i], scales)
            
            # Calculate average dynamics measures across variables
            avg_dynamics = self._average_dynamics_measures(dynamics_results)
            results['dynamics'] = {
                'individual': dynamics_results,
                'average': avg_dynamics
            }
        except Exception as e:
            warnings.warn(f"Dynamics analysis failed: {e}")
            results['dynamics'] = {'individual': {}, 'average': {}}
        
        # Network structure analysis
        try:
            network_results = self.network_analyzer.analyze(
                data, network_threshold, network_method, hypergraph_k
            )
            results['network_structure'] = network_results
        except Exception as e:
            warnings.warn(f"Network structure analysis failed: {e}")
            results['network_structure'] = {
                'modularity': np.nan,
                'num_nodes': data.shape[1],
                'num_edges': 0,
                'density': 0.0,
                'avg_clustering': 0.0,
                'assortativity': 0.0,
                'small_world_coeff': 0.0,
                'hypergraph_entropy': np.nan,
                'hyperedge_density': np.nan,
                'node_hyperedge_entropy': np.nan
            }
        
        # Calculate composite complexity score
        try:
            composite_score = self._calculate_composite_complexity(results)
            results['composite_complexity'] = composite_score
        except Exception as e:
            warnings.warn(f"Composite complexity calculation failed: {e}")
            results['composite_complexity'] = np.nan
        
        return results
    
    def _average_dynamics_measures(self, dynamics_results: Dict) -> Dict[str, Union[float, Dict]]:
        """
        Calculate average dynamics measures across multiple variables
        
        Args:
            dynamics_results: Dictionary containing dynamics results for each variable
            
        Returns:
            Dictionary containing averaged dynamics measures
        """
        if not dynamics_results:
            return {}
        
        # Get all variable results
        var_results = list(dynamics_results.values())
        
        # Initialize averaged results
        avg_results = {}
        
        # Average scalar measures
        scalar_measures = [
            'mse_mean', 'mse_std', 'mse_slope', 'dfa_alpha', 'hurst_exponent',
            'spectral_slope', 'power_law_exponent', 'criticality_index',
            'coefficient_of_variation', 'autocorrelation_lag1'
        ]
        
        for measure in scalar_measures:
            values = []
            for var_result in var_results:
                if measure in var_result and not np.isnan(var_result[measure]):
                    values.append(var_result[measure])
            
            if values:
                avg_results[measure] = np.mean(values)
            else:
                avg_results[measure] = np.nan
        
        # Average multiscale entropy curves
        if 'multiscale_entropy' in var_results[0]:
            all_scales = set()
            for var_result in var_results:
                if 'multiscale_entropy' in var_result:
                    all_scales.update(var_result['multiscale_entropy'].keys())
            
            avg_mse = {}
            for scale in sorted(all_scales):
                values = []
                for var_result in var_results:
                    if 'multiscale_entropy' in var_result and scale in var_result['multiscale_entropy']:
                        val = var_result['multiscale_entropy'][scale]
                        if not np.isnan(val):
                            values.append(val)
                
                if values:
                    avg_mse[scale] = np.mean(values)
                else:
                    avg_mse[scale] = np.nan
            
            avg_results['multiscale_entropy'] = avg_mse
        
        return avg_results
    
    def _calculate_composite_complexity(self, results: Dict) -> float:
        """
        Calculate a composite complexity score from all measures
        
        Args:
            results: Results from comprehensive analysis
            
        Returns:
            Composite complexity score
        """
        scores = []
        
        # Information theory scores
        if 'information_theory' in results:
            info = results['information_theory']
            if not np.isnan(info.get('synergy', np.nan)):
                scores.append(info['synergy'])
            if not np.isnan(info.get('phi', np.nan)):
                scores.append(info['phi'])
            if not np.isnan(info.get('multi_information', np.nan)):
                scores.append(info['multi_information'])
        
        # Dynamics scores
        if 'dynamics' in results and 'average' in results['dynamics']:
            dynamics = results['dynamics']['average']
            if not np.isnan(dynamics.get('mse_mean', np.nan)):
                scores.append(dynamics['mse_mean'])
            if not np.isnan(dynamics.get('dfa_alpha', np.nan)):
                # Normalize DFA alpha (typical range 0.5-2.0)
                normalized_alpha = min(1.0, max(0.0, (dynamics['dfa_alpha'] - 0.5) / 1.5))
                scores.append(normalized_alpha)
            if not np.isnan(dynamics.get('criticality_index', np.nan)):
                scores.append(dynamics['criticality_index'])
        
        # Network structure scores
        if 'network_structure' in results:
            network = results['network_structure']
            if not np.isnan(network.get('modularity', np.nan)):
                scores.append(network['modularity'])
            if not np.isnan(network.get('avg_clustering', np.nan)):
                scores.append(network['avg_clustering'])
            if not np.isnan(network.get('hypergraph_entropy', np.nan)):
                scores.append(network['hypergraph_entropy'])
        
        # Calculate composite score
        if scores:
            # Normalize and average all scores
            scores = np.array(scores)
            # Simple normalization (assumes scores are roughly in [0, 1] range)
            normalized_scores = np.clip(scores, 0, 1)
            composite_score = np.mean(normalized_scores)
        else:
            composite_score = np.nan
        
        return composite_score
    
    def compare_systems(self, data_list: List[np.ndarray], 
                       system_names: Optional[List[str]] = None,
                       **kwargs) -> pd.DataFrame:
        """
        Compare complexity across multiple systems
        
        Args:
            data_list: List of multivariate time series data
            system_names: Names for each system (if None, uses 'System_0', 'System_1', etc.)
            **kwargs: Additional arguments passed to analyze_multivariate
            
        Returns:
            DataFrame with complexity measures for each system
        """
        if system_names is None:
            system_names = [f'System_{i}' for i in range(len(data_list))]
        
        if len(data_list) != len(system_names):
            raise ValueError("Number of data arrays must match number of system names")
        
        all_results = []
        
        for i, (data, name) in enumerate(zip(data_list, system_names)):
            print(f"Analyzing {name}...")
            
            try:
                results = self.analyze_multivariate(data, **kwargs)
                
                # Flatten results for DataFrame
                flattened = self._flatten_results(results)
                flattened['system_name'] = name
                flattened['system_index'] = i
                
                all_results.append(flattened)
                
            except Exception as e:
                warnings.warn(f"Failed to analyze {name}: {e}")
                # Create empty result
                flattened = {'system_name': name, 'system_index': i}
                all_results.append(flattened)
        
        return pd.DataFrame(all_results)
    
    def _flatten_results(self, results: Dict, prefix: str = '') -> Dict:
        """
        Flatten nested results dictionary for DataFrame creation
        
        Args:
            results: Nested results dictionary
            prefix: Prefix for keys (used for recursion)
            
        Returns:
            Flattened dictionary
        """
        flattened = {}
        
        for key, value in results.items():
            new_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                nested = self._flatten_results(value, f"{new_key}_")
                flattened.update(nested)
            else:
                flattened[new_key] = value
        
        return flattened
    
    def get_complexity_summary(self, results: Dict) -> Dict[str, Union[str, float]]:
        """
        Generate a human-readable summary of complexity analysis
        
        Args:
            results: Results from analyze_multivariate
            
        Returns:
            Dictionary containing complexity summary
        """
        summary = {}
        
        # Information theory summary
        if 'information_theory' in results:
            info = results['information_theory']
            summary['synergy_level'] = self._categorize_value(info.get('synergy', 0), [0, 0.1, 0.5], ['Low', 'Medium', 'High'])
            summary['integration_level'] = self._categorize_value(info.get('phi', 0), [0, 0.1, 0.3], ['Low', 'Medium', 'High'])
            summary['correlation_level'] = self._categorize_value(info.get('multi_information', 0), [0, 0.2, 0.5], ['Low', 'Medium', 'High'])
        
        # Dynamics summary
        if 'dynamics' in results and 'average' in results['dynamics']:
            dynamics = results['dynamics']['average']
            summary['temporal_complexity'] = self._categorize_value(dynamics.get('mse_mean', 0), [0, 0.5, 1.0], ['Low', 'Medium', 'High'])
            summary['fractal_nature'] = self._categorize_value(dynamics.get('dfa_alpha', 0.5), [0.5, 1.0, 1.5], ['Anti-persistent', 'Random', 'Persistent'])
            summary['criticality_level'] = self._categorize_value(dynamics.get('criticality_index', 0), [0, 0.5, 1.0], ['Sub-critical', 'Near-critical', 'Critical'])
        
        # Network structure summary
        if 'network_structure' in results:
            network = results['network_structure']
            summary['modularity_level'] = self._categorize_value(network.get('modularity', 0), [0, 0.3, 0.7], ['Low', 'Medium', 'High'])
            summary['connectivity_level'] = self._categorize_value(network.get('density', 0), [0, 0.1, 0.5], ['Sparse', 'Moderate', 'Dense'])
            summary['clustering_level'] = self._categorize_value(network.get('avg_clustering', 0), [0, 0.3, 0.7], ['Low', 'Medium', 'High'])
        
        # Overall complexity
        if 'composite_complexity' in results:
            composite = results['composite_complexity']
            summary['overall_complexity'] = self._categorize_value(composite, [0, 0.3, 0.7], ['Simple', 'Moderately Complex', 'Highly Complex'])
        
        return summary
    
    def _categorize_value(self, value: float, thresholds: List[float], labels: List[str]) -> str:
        """
        Categorize a numerical value into discrete levels
        
        Args:
            value: Numerical value to categorize
            thresholds: Threshold values for categorization
            labels: Labels for each category
            
        Returns:
            Category label
        """
        if np.isnan(value):
            return 'Unknown'
        
        for i, threshold in enumerate(thresholds):
            if value <= threshold:
                return labels[i]
        
        return labels[-1]
