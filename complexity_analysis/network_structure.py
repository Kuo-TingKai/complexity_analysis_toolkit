"""
Network Structure Analysis Module

This module implements network-based complexity measures:
- Modularity: measure of community structure in networks
- Hypergraph entropy: entropy measures for hypergraph structures
- Additional network complexity measures
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import itertools
import warnings
from typing import List, Tuple, Dict, Optional, Union


class NetworkStructureAnalyzer:
    """
    Analyzer for network structure complexity measures
    """
    
    def __init__(self, n_clusters: int = None, clustering_method: str = 'spectral'):
        """
        Initialize the network structure analyzer
        
        Args:
            n_clusters: Number of clusters for modularity (if None, auto-detect)
            clustering_method: Method for community detection ('spectral', 'hierarchical')
        """
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
    
    def _create_network_from_data(self, data: np.ndarray, threshold: Optional[float] = None, 
                                 method: str = 'correlation') -> nx.Graph:
        """
        Create network from multivariate data
        
        Args:
            data: Multivariate time series data (rows=time, cols=variables)
            threshold: Threshold for edge creation (if None, uses adaptive threshold)
            method: Method for edge weight calculation ('correlation', 'mutual_info', 'distance')
            
        Returns:
            NetworkX graph object
        """
        n_vars = data.shape[1]
        
        if method == 'correlation':
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(data.T)
            # Remove diagonal and take absolute values
            weights = np.abs(corr_matrix)
            np.fill_diagonal(weights, 0)
            
        elif method == 'distance':
            # Calculate distance matrix (1 - correlation)
            corr_matrix = np.corrcoef(data.T)
            weights = 1 - np.abs(corr_matrix)
            np.fill_diagonal(weights, 0)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Determine threshold if not provided
        if threshold is None:
            # Use adaptive threshold (keep top 20% of connections)
            flat_weights = weights[weights > 0]
            if len(flat_weights) > 0:
                threshold = np.percentile(flat_weights, 80)
            else:
                threshold = 0.5
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for i in range(n_vars):
            G.add_node(i)
        
        # Add edges based on threshold
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if method == 'correlation' and weights[i, j] >= threshold:
                    G.add_edge(i, j, weight=weights[i, j])
                elif method == 'distance' and weights[i, j] <= threshold:
                    G.add_edge(i, j, weight=1/weights[i, j])  # Invert distance for weight
        
        return G
    
    def _detect_communities(self, G: nx.Graph) -> List[List[int]]:
        """
        Detect communities in the network
        
        Args:
            G: NetworkX graph
            
        Returns:
            List of communities (each community is a list of node indices)
        """
        if G.number_of_nodes() == 0:
            return []
        
        # Get adjacency matrix
        adj_matrix = nx.adjacency_matrix(G).toarray()
        
        if self.clustering_method == 'spectral':
            # Use spectral clustering
            if self.n_clusters is None:
                # Estimate number of clusters using eigenvalue gap
                eigenvals = np.linalg.eigvals(adj_matrix)
                eigenvals = np.sort(eigenvals)[::-1]
                eigen_gaps = np.diff(eigenvals)
                if len(eigen_gaps) > 0:
                    self.n_clusters = np.argmax(eigen_gaps) + 2
                else:
                    self.n_clusters = 2
            
            # Ensure n_clusters doesn't exceed number of nodes
            self.n_clusters = min(self.n_clusters, G.number_of_nodes())
            
            if self.n_clusters < 2:
                return [list(G.nodes())]
            
            clustering = SpectralClustering(n_clusters=self.n_clusters, 
                                          affinity='precomputed',
                                          random_state=42)
            labels = clustering.fit_predict(adj_matrix)
            
        elif self.clustering_method == 'hierarchical':
            # Use hierarchical clustering
            if self.n_clusters is None:
                self.n_clusters = max(2, G.number_of_nodes() // 3)
            
            clustering = AgglomerativeClustering(n_clusters=self.n_clusters,
                                               affinity='precomputed',
                                               linkage='average')
            labels = clustering.fit_predict(1 - adj_matrix)  # Convert to distance
            
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # Group nodes by community
        communities = [[] for _ in range(self.n_clusters)]
        for node, label in enumerate(labels):
            communities[label].append(node)
        
        # Remove empty communities
        communities = [comm for comm in communities if comm]
        
        return communities
    
    def modularity(self, G: nx.Graph, communities: Optional[List[List[int]]] = None) -> float:
        """
        Calculate modularity of the network
        
        Args:
            G: NetworkX graph
            communities: Pre-computed communities (if None, will detect automatically)
            
        Returns:
            Modularity value
        """
        if G.number_of_nodes() == 0:
            return 0.0
        
        if communities is None:
            communities = self._detect_communities(G)
        
        if not communities:
            return 0.0
        
        # Calculate total edge weight
        total_weight = G.size(weight='weight') if G.size() > 0 else 1.0
        
        if total_weight == 0:
            return 0.0
        
        modularity = 0.0
        
        for community in communities:
            if not community:
                continue
                
            # Internal connections
            internal_weight = 0.0
            for node in community:
                for neighbor in G.neighbors(node):
                    if neighbor in community:
                        internal_weight += G[node][neighbor].get('weight', 1.0)
            
            # Degree sum for community
            degree_sum = sum(G.degree(node, weight='weight') for node in community)
            
            # Modularity contribution
            modularity += (internal_weight / total_weight) - (degree_sum / (2 * total_weight))**2
        
        return modularity
    
    def _create_hypergraph(self, data: np.ndarray, k: int = 3) -> List[List[int]]:
        """
        Create hypergraph from multivariate data
        
        Args:
            data: Multivariate time series data
            k: Size of hyperedges (number of variables in each hyperedge)
            
        Returns:
            List of hyperedges (each hyperedge is a list of variable indices)
        """
        n_vars = data.shape[1]
        hyperedges = []
        
        # Create hyperedges based on high-order correlations
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                for l in range(j + 1, min(j + k - 1, n_vars)):
                    # Calculate correlation between variables i, j, l
                    subset = data[:, [i, j, l]]
                    corr_matrix = np.corrcoef(subset.T)
                    
                    # Check if all pairwise correlations are above threshold
                    min_corr = np.min(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                    
                    if min_corr > 0.3:  # Threshold for hyperedge creation
                        hyperedges.append([i, j, l])
        
        return hyperedges
    
    def hypergraph_entropy(self, data: np.ndarray, k: int = 3) -> Dict[str, float]:
        """
        Calculate hypergraph entropy measures
        
        Args:
            data: Multivariate time series data
            k: Size of hyperedges for hypergraph construction
            
        Returns:
            Dictionary containing hypergraph entropy measures
        """
        results = {}
        
        try:
            # Create hypergraph
            hyperedges = self._create_hypergraph(data, k)
            
            if not hyperedges:
                results['hypergraph_entropy'] = 0.0
                results['hyperedge_density'] = 0.0
                results['node_hyperedge_entropy'] = 0.0
                return results
            
            # Calculate hypergraph entropy
            n_nodes = data.shape[1]
            n_hyperedges = len(hyperedges)
            
            # Node degree in hypergraph (number of hyperedges each node participates in)
            node_degrees = [0] * n_nodes
            for hyperedge in hyperedges:
                for node in hyperedge:
                    node_degrees[node] += 1
            
            # Hypergraph entropy (entropy of node degrees)
            if sum(node_degrees) > 0:
                degree_probs = np.array(node_degrees) / sum(node_degrees)
                degree_probs = degree_probs[degree_probs > 0]  # Remove zeros
                hypergraph_entropy = -np.sum(degree_probs * np.log2(degree_probs))
            else:
                hypergraph_entropy = 0.0
            
            results['hypergraph_entropy'] = hypergraph_entropy
            
            # Hyperedge density
            max_possible_hyperedges = len(list(itertools.combinations(range(n_nodes), k)))
            hyperedge_density = n_hyperedges / max_possible_hyperedges if max_possible_hyperedges > 0 else 0
            results['hyperedge_density'] = hyperedge_density
            
            # Node-hyperedge entropy (entropy of hyperedge sizes)
            hyperedge_sizes = [len(hyperedge) for hyperedge in hyperedges]
            if hyperedge_sizes:
                size_probs = np.bincount(hyperedge_sizes) / len(hyperedge_sizes)
                size_probs = size_probs[size_probs > 0]
                node_hyperedge_entropy = -np.sum(size_probs * np.log2(size_probs))
            else:
                node_hyperedge_entropy = 0.0
            
            results['node_hyperedge_entropy'] = node_hyperedge_entropy
            
        except Exception as e:
            warnings.warn(f"Failed to calculate hypergraph entropy: {e}")
            results['hypergraph_entropy'] = np.nan
            results['hyperedge_density'] = np.nan
            results['node_hyperedge_entropy'] = np.nan
        
        return results
    
    def network_complexity_measures(self, G: nx.Graph) -> Dict[str, float]:
        """
        Calculate additional network complexity measures
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary containing network complexity measures
        """
        results = {}
        
        try:
            # Basic network statistics
            results['num_nodes'] = G.number_of_nodes()
            results['num_edges'] = G.number_of_edges()
            
            if G.number_of_nodes() == 0:
                results['density'] = 0.0
                results['avg_clustering'] = 0.0
                results['assortativity'] = 0.0
                return results
            
            # Network density
            max_edges = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
            results['density'] = G.number_of_edges() / max_edges if max_edges > 0 else 0.0
            
            # Average clustering coefficient
            if G.number_of_nodes() > 2:
                results['avg_clustering'] = nx.average_clustering(G)
            else:
                results['avg_clustering'] = 0.0
            
            # Assortativity (degree correlation)
            if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
                try:
                    results['assortativity'] = nx.degree_assortativity_coefficient(G)
                except:
                    results['assortativity'] = 0.0
            else:
                results['assortativity'] = 0.0
            
            # Small-world coefficient
            if G.number_of_nodes() > 2:
                try:
                    # Calculate average shortest path length
                    if nx.is_connected(G):
                        avg_path_length = nx.average_shortest_path_length(G)
                    else:
                        # For disconnected graphs, use largest component
                        largest_cc = max(nx.connected_components(G), key=len)
                        subgraph = G.subgraph(largest_cc)
                        avg_path_length = nx.average_shortest_path_length(subgraph)
                    
                    # Calculate clustering coefficient
                    clustering = nx.average_clustering(G)
                    
                    # Small-world coefficient (simplified)
                    results['small_world_coeff'] = clustering / (avg_path_length + 1e-10)
                    
                except:
                    results['small_world_coeff'] = 0.0
            else:
                results['small_world_coeff'] = 0.0
            
        except Exception as e:
            warnings.warn(f"Failed to calculate network complexity measures: {e}")
            results.update({
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': 0.0,
                'avg_clustering': 0.0,
                'assortativity': 0.0,
                'small_world_coeff': 0.0
            })
        
        return results
    
    def analyze(self, data: np.ndarray, threshold: Optional[float] = None, 
                method: str = 'correlation', k: int = 3) -> Dict[str, Union[float, Dict]]:
        """
        Perform comprehensive network structure analysis
        
        Args:
            data: Multivariate time series data (rows=time, cols=variables)
            threshold: Threshold for network construction (if None, uses adaptive)
            method: Method for edge weight calculation
            k: Size of hyperedges for hypergraph analysis
            
        Returns:
            Dictionary containing all network structure measures
        """
        results = {}
        
        try:
            # Create network from data
            G = self._create_network_from_data(data, threshold, method)
            
            # Calculate modularity
            modularity_val = self.modularity(G)
            results['modularity'] = modularity_val
            
            # Additional network complexity measures
            network_measures = self.network_complexity_measures(G)
            results.update(network_measures)
            
        except Exception as e:
            warnings.warn(f"Failed to perform network analysis: {e}")
            results['modularity'] = np.nan
            results.update({
                'num_nodes': data.shape[1],
                'num_edges': 0,
                'density': 0.0,
                'avg_clustering': 0.0,
                'assortativity': 0.0,
                'small_world_coeff': 0.0
            })
        
        try:
            # Hypergraph entropy measures
            hypergraph_results = self.hypergraph_entropy(data, k)
            results.update(hypergraph_results)
            
        except Exception as e:
            warnings.warn(f"Failed to perform hypergraph analysis: {e}")
            results.update({
                'hypergraph_entropy': np.nan,
                'hyperedge_density': np.nan,
                'node_hyperedge_entropy': np.nan
            })
        
        return results
