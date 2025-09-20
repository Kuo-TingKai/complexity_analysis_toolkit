#!/usr/bin/env python3
"""
Advanced Example: Real-world Data Analysis

This example demonstrates how to use the complexity analysis toolkit on
real-world datasets and perform comparative analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from complexity_analysis import ComplexityAnalyzer

# Set random seed for reproducibility
np.random.seed(42)

def load_financial_data():
    """
    Generate synthetic financial market data with different regimes
    
    Returns:
        Dictionary containing different market regimes
    """
    n_points = 1000
    n_assets = 5
    
    regimes = {}
    
    # 1. Bull market (trending upward with low volatility)
    t = np.linspace(0, 10, n_points)
    bull_market = np.zeros((n_points, n_assets))
    for i in range(n_assets):
        bull_market[:, i] = 100 + 2*t + 0.5*np.sin(t + i) + np.random.randn(n_points) * 0.1
    
    regimes['bull_market'] = bull_market
    
    # 2. Bear market (trending downward with high volatility)
    bear_market = np.zeros((n_points, n_assets))
    for i in range(n_assets):
        bear_market[:, i] = 100 - 1.5*t + 0.3*np.cos(t + i) + np.random.randn(n_points) * 0.5
    
    regimes['bear_market'] = bear_market
    
    # 3. Sideways market (ranging with moderate volatility)
    sideways_market = np.zeros((n_points, n_assets))
    for i in range(n_assets):
        sideways_market[:, i] = 100 + 0.5*np.sin(0.5*t + i) + np.random.randn(n_points) * 0.3
    
    regimes['sideways_market'] = sideways_market
    
    # 4. Crisis market (highly volatile, correlated crashes)
    crisis_market = np.zeros((n_points, n_assets))
    crisis_market[0, :] = 100
    
    for i in range(1, n_points):
        # High correlation during crisis
        shock = np.random.randn() * 2  # Large shock
        for j in range(n_assets):
            # Assets move together during crisis
            crisis_market[i, j] = crisis_market[i-1, j] * (1 + 0.001*shock + 0.1*np.random.randn())
    
    regimes['crisis_market'] = crisis_market
    
    return regimes

def load_ecological_data():
    """
    Generate synthetic ecological system data with different stability regimes
    
    Returns:
        Dictionary containing different ecological systems
    """
    n_points = 1000
    n_species = 4
    
    systems = {}
    
    # 1. Stable ecosystem (low variability, high resilience)
    stable_ecosystem = np.zeros((n_points, n_species))
    for i in range(n_species):
        stable_ecosystem[:, i] = 50 + 5*np.sin(0.1*np.arange(n_points) + i) + np.random.randn(n_points) * 0.5
    
    systems['stable_ecosystem'] = stable_ecosystem
    
    # 2. Oscillating ecosystem (periodic dynamics)
    oscillating_ecosystem = np.zeros((n_points, n_species))
    for i in range(n_species):
        period = 50 + i * 10
        oscillating_ecosystem[:, i] = 50 + 20*np.sin(2*np.pi*np.arange(n_points)/period + i) + np.random.randn(n_points) * 1
    
    systems['oscillating_ecosystem'] = oscillating_ecosystem
    
    # 3. Chaotic ecosystem (complex dynamics)
    chaotic_ecosystem = np.zeros((n_points, n_species))
    chaotic_ecosystem[0, :] = np.random.randn(n_species) * 10 + 50
    
    for i in range(1, n_points):
        for j in range(n_species):
            # Lotka-Volterra-like chaotic dynamics
            x = chaotic_ecosystem[i-1, j] / 50  # Normalize
            y = chaotic_ecosystem[i-1, (j+1)%n_species] / 50
            
            dx = 0.1 * x * (1 - x) - 0.05 * x * y
            chaotic_ecosystem[i, j] = max(0, chaotic_ecosystem[i-1, j] + dx * 50)
    
    systems['chaotic_ecosystem'] = chaotic_ecosystem
    
    # 4. Collapsing ecosystem (declining populations)
    collapsing_ecosystem = np.zeros((n_points, n_species))
    collapsing_ecosystem[0, :] = 100
    
    for i in range(1, n_points):
        for j in range(n_species):
            # Exponential decline with noise
            decline_rate = 0.001 + 0.0005 * j  # Different decline rates for different species
            collapsing_ecosystem[i, j] = max(0, collapsing_ecosystem[i-1, j] * (1 - decline_rate) + np.random.randn() * 0.5)
    
    systems['collapsing_ecosystem'] = collapsing_ecosystem
    
    return systems

def analyze_regime_changes(data_dict, regime_name):
    """
    Analyze complexity changes within a single regime over time
    
    Args:
        data_dict: Dictionary containing regime data
        regime_name: Name of the regime to analyze
        
    Returns:
        Dictionary containing time-varying complexity measures
    """
    data = data_dict[regime_name]
    window_size = 200
    step_size = 50
    
    # Calculate rolling complexity measures
    complexity_evolution = []
    time_windows = []
    
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        window_data = data[start:end]
        
        # Analyze this window
        analyzer = ComplexityAnalyzer()
        window_results = analyzer.analyze_multivariate(window_data)
        
        complexity_evolution.append(window_results['composite_complexity'])
        time_windows.append((start + end) / 2)  # Midpoint of window
    
    return {
        'time_windows': time_windows,
        'complexity_evolution': complexity_evolution,
        'data': data
    }

def compare_regimes(data_dict, regime_names, analyzer):
    """
    Compare complexity across different regimes
    
    Args:
        data_dict: Dictionary containing regime data
        regime_names: List of regime names to compare
        analyzer: ComplexityAnalyzer instance
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for regime_name in regime_names:
        if regime_name in data_dict:
            print(f"Analyzing {regime_name}...")
            
            # Analyze the regime
            results = analyzer.analyze_multivariate(data_dict[regime_name])
            
            # Extract key measures
            info = results['information_theory']
            dynamics = results['dynamics']['average']
            network = results['network_structure']
            
            comparison_data.append({
                'Regime': regime_name,
                'Composite_Complexity': results['composite_complexity'],
                'Synergy': info['synergy'],
                'Phi': info['phi'],
                'Multi_Information': info['multi_information'],
                'MSE_Mean': dynamics.get('mse_mean', np.nan),
                'DFA_Alpha': dynamics.get('dfa_alpha', np.nan),
                'Criticality_Index': dynamics.get('criticality_index', np.nan),
                'Modularity': network['modularity'],
                'Network_Density': network['density'],
                'Avg_Clustering': network['avg_clustering'],
                'Hypergraph_Entropy': network['hypergraph_entropy']
            })
    
    return pd.DataFrame(comparison_data)

def create_comparison_visualizations(comparison_df, regime_evolution_data):
    """
    Create comprehensive visualizations for regime comparison
    
    Args:
        comparison_df: DataFrame with regime comparison results
        regime_evolution_data: Dictionary with time evolution data
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Regime Complexity Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Composite complexity comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(comparison_df['Regime'], comparison_df['Composite_Complexity'], 
                   color=['green', 'red', 'orange', 'purple'], alpha=0.7)
    ax1.set_title('Composite Complexity by Regime')
    ax1.set_ylabel('Complexity Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, comparison_df['Composite_Complexity']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 2: Information theory measures heatmap
    ax2 = axes[0, 1]
    info_measures = ['Synergy', 'Phi', 'Multi_Information']
    info_data = comparison_df[['Regime'] + info_measures].set_index('Regime')
    sns.heatmap(info_data.T, annot=True, cmap='YlOrRd', ax=ax2, fmt='.3f')
    ax2.set_title('Information Theory Measures')
    
    # Plot 3: Dynamics measures heatmap
    ax3 = axes[0, 2]
    dynamics_measures = ['MSE_Mean', 'DFA_Alpha', 'Criticality_Index']
    dynamics_data = comparison_df[['Regime'] + dynamics_measures].set_index('Regime')
    sns.heatmap(dynamics_data.T, annot=True, cmap='Blues', ax=ax3, fmt='.3f')
    ax3.set_title('Dynamics Measures')
    
    # Plot 4: Network structure measures
    ax4 = axes[1, 0]
    network_measures = ['Modularity', 'Network_Density', 'Avg_Clustering', 'Hypergraph_Entropy']
    x = np.arange(len(comparison_df))
    width = 0.2
    
    for i, measure in enumerate(network_measures):
        ax4.bar(x + i*width, comparison_df[measure], width, label=measure, alpha=0.8)
    
    ax4.set_title('Network Structure Measures')
    ax4.set_ylabel('Value')
    ax4.set_xlabel('Regimes')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(comparison_df['Regime'], rotation=45)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 5: Complexity evolution over time (financial markets)
    ax5 = axes[1, 1]
    if 'financial' in regime_evolution_data:
        for regime_name, evolution in regime_evolution_data['financial'].items():
            ax5.plot(evolution['time_windows'], evolution['complexity_evolution'], 
                    'o-', label=regime_name, alpha=0.7)
    ax5.set_title('Financial Market Complexity Evolution')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Complexity')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Complexity evolution over time (ecological systems)
    ax6 = axes[1, 2]
    if 'ecological' in regime_evolution_data:
        for regime_name, evolution in regime_evolution_data['ecological'].items():
            ax6.plot(evolution['time_windows'], evolution['complexity_evolution'], 
                    'o-', label=regime_name, alpha=0.7)
    ax6.set_title('Ecological System Complexity Evolution')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Complexity')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Scatter plot of complexity vs stability
    ax7 = axes[2, 0]
    # Use criticality index as stability measure
    stability = comparison_df['Criticality_Index']
    complexity = comparison_df['Composite_Complexity']
    
    scatter = ax7.scatter(stability, complexity, 
                         c=range(len(comparison_df)), 
                         cmap='viridis', s=100, alpha=0.7)
    
    for i, regime in enumerate(comparison_df['Regime']):
        ax7.annotate(regime, (stability.iloc[i], complexity.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax7.set_xlabel('Criticality Index (Stability)')
    ax7.set_ylabel('Composite Complexity')
    ax7.set_title('Complexity vs Stability')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Correlation matrix of all measures
    ax8 = axes[2, 1]
    measures_cols = [col for col in comparison_df.columns if col != 'Regime']
    correlation_matrix = comparison_df[measures_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                ax=ax8, fmt='.2f', square=True)
    ax8.set_title('Measures Correlation Matrix')
    
    # Plot 9: Complexity ranking
    ax9 = axes[2, 2]
    sorted_df = comparison_df.sort_values('Composite_Complexity', ascending=True)
    bars = ax9.barh(sorted_df['Regime'], sorted_df['Composite_Complexity'], 
                    color=['green', 'orange', 'red', 'purple'], alpha=0.7)
    ax9.set_title('Complexity Ranking')
    ax9.set_xlabel('Composite Complexity Score')
    
    # Add value labels
    for bar, score in zip(bars, sorted_df['Composite_Complexity']):
        ax9.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('advanced_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the advanced complexity analysis example
    """
    print("Complexity Analysis Toolkit - Advanced Example")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = ComplexityAnalyzer(
            info_binning_method='quantile',
            info_n_bins=8,
            dynamics_tolerance=0.15,
            dynamics_max_scale=15
        )
        
        # Load and analyze financial data
        print("Loading financial market data...")
        financial_regimes = load_financial_data()
        financial_names = list(financial_regimes.keys())
        
        # Analyze complexity evolution over time
        print("Analyzing complexity evolution in financial markets...")
        financial_evolution = {}
        for regime_name in financial_names:
            financial_evolution[regime_name] = analyze_regime_changes(financial_regimes, regime_name)
        
        # Load and analyze ecological data
        print("Loading ecological system data...")
        ecological_systems = load_ecological_data()
        ecological_names = list(ecological_systems.keys())
        
        # Analyze complexity evolution over time
        print("Analyzing complexity evolution in ecological systems...")
        ecological_evolution = {}
        for regime_name in ecological_names:
            ecological_evolution[regime_name] = analyze_regime_changes(ecological_systems, regime_name)
        
        # Compare all regimes
        print("Comparing complexity across all regimes...")
        all_regimes = {**financial_regimes, **ecological_systems}
        all_names = financial_names + ecological_names
        
        comparison_df = compare_regimes(all_regimes, all_names, analyzer)
        
        # Print results
        print("\n=== Regime Comparison Results ===")
        print(comparison_df.round(4))
        
        # Create visualizations
        print("Creating advanced visualizations...")
        regime_evolution_data = {
            'financial': financial_evolution,
            'ecological': ecological_evolution
        }
        create_comparison_visualizations(comparison_df, regime_evolution_data)
        
        # Print insights
        print("\n=== Key Insights ===")
        highest_complexity = comparison_df.loc[comparison_df['Composite_Complexity'].idxmax()]
        lowest_complexity = comparison_df.loc[comparison_df['Composite_Complexity'].idxmin()]
        
        print(f"Highest Complexity Regime: {highest_complexity['Regime']} (Score: {highest_complexity['Composite_Complexity']:.3f})")
        print(f"Lowest Complexity Regime: {lowest_complexity['Regime']} (Score: {lowest_complexity['Composite_Complexity']:.3f})")
        
        # Analyze correlations
        complexity_stability_corr = comparison_df['Composite_Complexity'].corr(comparison_df['Criticality_Index'])
        print(f"Complexity-Stability Correlation: {complexity_stability_corr:.3f}")
        
        print("Advanced analysis complete! Results saved as 'advanced_complexity_analysis.png'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
