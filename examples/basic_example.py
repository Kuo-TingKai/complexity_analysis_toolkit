#!/usr/bin/env python3
"""
Basic Example: Complexity Analysis of Synthetic Systems

This example demonstrates how to use the complexity analysis toolkit to analyze
different types of synthetic systems and compare their complexity measures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from complexity_analysis import ComplexityAnalyzer

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_systems():
    """
    Generate synthetic systems with different complexity levels
    
    Returns:
        Dictionary containing different synthetic systems
    """
    n_points = 1000
    n_vars = 5
    
    systems = {}
    
    # 1. Random system (low complexity) - Pure white noise
    systems['random'] = np.random.randn(n_points, n_vars)
    
    # 2. Linear system (medium complexity) - Linear trend with noise
    t = np.linspace(0, 10, n_points)
    linear_system = np.zeros((n_points, n_vars))
    for i in range(n_vars):
        linear_system[:, i] = 0.5 * t + np.random.randn(n_points) * 0.1
    systems['linear'] = linear_system
    
    # 3. Coupled oscillator system (high complexity) - True oscillatory behavior
    dt = 0.01
    t = np.arange(0, n_points * dt, dt)
    coupled_system = np.zeros((len(t), n_vars))
    
    # Initialize with different phases for each oscillator
    for j in range(n_vars):
        coupled_system[0, j] = np.sin(2 * np.pi * j / n_vars)  # Different initial phases
    
    # Simulate coupled oscillators with proper oscillatory dynamics
    for i in range(1, len(t)):
        for j in range(n_vars):
            # Natural frequency with coupling to neighbors
            omega = 1.0 + 0.1 * j  # Different natural frequencies
            
            # Coupling with neighbors (bidirectional)
            coupling = 0
            if j > 0:
                coupling += 0.2 * np.sin(coupled_system[i-1, j-1] - coupled_system[i-1, j])
            if j < n_vars - 1:
                coupling += 0.2 * np.sin(coupled_system[i-1, j+1] - coupled_system[i-1, j])
            
            # Update oscillator phase
            coupled_system[i, j] = coupled_system[i-1, j] + dt * (omega + coupling)
    
    # Convert phases to actual oscillator values
    for j in range(n_vars):
        coupled_system[:, j] = np.sin(coupled_system[:, j]) + 0.1 * np.random.randn(len(t))
    
    systems['coupled_oscillators'] = coupled_system
    
    # 4. Chaotic system (very high complexity) - True chaotic dynamics
    chaotic_system = np.zeros((n_points, n_vars))
    
    # Initialize Lorenz-like chaotic system
    x = np.random.randn() * 0.1
    y = np.random.randn() * 0.1
    z = np.random.randn() * 0.1 + 25
    
    dt = 0.01
    
    for i in range(n_points):
        # Lorenz equations
        dx = 10 * (y - x)
        dy = x * (28 - z) - y
        dz = x * y - (8/3) * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
        # Store different components for different variables
        chaotic_system[i, 0] = x
        chaotic_system[i, 1] = y
        chaotic_system[i, 2] = z
        chaotic_system[i, 3] = x + y  # Combined signal
        chaotic_system[i, 4] = np.sqrt(x**2 + y**2 + z**2)  # Magnitude
    
    systems['chaotic'] = chaotic_system
    
    return systems


def visualize_results(systems, results):
    """
    Create visualizations of the analysis results
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Complexity Analysis of Synthetic Systems', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series of each system
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green', 'purple']
    for i, (name, data) in enumerate(systems.items()):
        ax1.plot(data[:200, 0], label=name, alpha=0.7, color=colors[i])
    ax1.set_title('Time Series (First Variable)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Composite complexity scores
    ax2 = axes[0, 1]
    complexity_scores = [results[name]['composite_complexity'] for name in systems.keys()]
    system_names = list(systems.keys())
    
    bars = ax2.bar(system_names, complexity_scores, color=['red', 'orange', 'green', 'blue'])
    ax2.set_title('Composite Complexity Scores')
    ax2.set_ylabel('Complexity Score')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, complexity_scores):
        if not np.isnan(score):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 3: Information theory measures
    ax3 = axes[0, 2]
    info_measures = ['synergy', 'phi', 'multi_information']
    x = np.arange(len(system_names))
    width = 0.25
    
    for i, measure in enumerate(info_measures):
        values = [results[name]['information_theory'][measure] for name in system_names]
        ax3.bar(x + i*width, values, width, label=measure, alpha=0.8)
    
    ax3.set_title('Information Theory Measures')
    ax3.set_ylabel('Value')
    ax3.set_xlabel('Systems')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(system_names)
    ax3.legend()
    
    # Plot 4: Dynamics measures
    ax4 = axes[1, 0]
    dynamics_measures = ['mse_mean', 'dfa_alpha', 'criticality_index']
    x = np.arange(len(system_names))
    
    for i, measure in enumerate(dynamics_measures):
        values = [results[name]['dynamics']['average'].get(measure, np.nan) for name in system_names]
        ax4.bar(x + i*width, values, width, label=measure, alpha=0.8)
    
    ax4.set_title('Dynamics Measures')
    ax4.set_ylabel('Value')
    ax4.set_xlabel('Systems')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(system_names)
    ax4.legend()
    
    # Plot 5: Network structure measures
    ax5 = axes[1, 1]
    network_measures = ['modularity', 'density', 'avg_clustering']
    x = np.arange(len(system_names))
    
    for i, measure in enumerate(network_measures):
        values = [results[name]['network_structure'].get(measure, np.nan) for name in system_names]
        ax5.bar(x + i*width, values, width, label=measure, alpha=0.8)
    
    ax5.set_title('Network Structure Measures')
    ax5.set_ylabel('Value')
    ax5.set_xlabel('Systems')
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(system_names)
    ax5.legend()
    
    # Plot 6: Multiscale entropy curves
    ax6 = axes[1, 2]
    for name in system_names:
        mse_data = results[name]['dynamics']['average'].get('multiscale_entropy', {})
        if mse_data:
            scales = list(mse_data.keys())
            entropies = list(mse_data.values())
            ax6.plot(scales, entropies, 'o-', label=name, alpha=0.7)
    
    ax6.set_title('Multiscale Entropy Curves')
    ax6.set_xlabel('Scale')
    ax6.set_ylabel('Sample Entropy')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complexity_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(results):
    """
    Print detailed analysis results
    """
    print("\n=== Detailed Analysis Results ===\n")
    
    for name, result in results.items():
        print(f"--- {name.upper()} SYSTEM ---")
        
        # Information theory
        info = result['information_theory']
        print(f"Information Theory:")
        print(f"  Synergy: {info['synergy']:.4f}")
        print(f"  Phi (Φ): {info['phi']:.4f}")
        print(f"  Multi-information: {info['multi_information']:.4f}")
        
        # Dynamics
        dynamics = result['dynamics']['average']
        print(f"Dynamics:")
        print(f"  MSE Mean: {dynamics.get('mse_mean', np.nan):.4f}")
        print(f"  DFA Alpha: {dynamics.get('dfa_alpha', np.nan):.4f}")
        print(f"  Criticality Index: {dynamics.get('criticality_index', np.nan):.4f}")
        print(f"  Hurst Exponent: {dynamics.get('hurst_exponent', np.nan):.4f}")
        
        # Network structure
        network = result['network_structure']
        print(f"Network Structure:")
        print(f"  Modularity: {network['modularity']:.4f}")
        print(f"  Density: {network['density']:.4f}")
        print(f"  Avg Clustering: {network['avg_clustering']:.4f}")
        print(f"  Hypergraph Entropy: {network['hypergraph_entropy']:.4f}")
        
        # Composite score
        print(f"Composite Complexity: {result['composite_complexity']:.4f}")
        print()

def main():
    """
    Main function to run the complexity analysis example
    """
    print("Complexity Analysis Toolkit - Basic Example")
    print("=" * 50)
    
    try:
        # Generate synthetic systems
        systems = generate_synthetic_systems()
        
        # Initialize complexity analyzer
        analyzer = ComplexityAnalyzer(
            info_binning_method='uniform',
            info_n_bins=10,
            dynamics_tolerance=0.2,
            dynamics_max_scale=10
        )
        
        # Analyze each system
        results = {}
        for name, data in systems.items():
            print(f"Analyzing {name} system...")
            
            # Perform comprehensive analysis
            analysis_results = analyzer.analyze_multivariate(data)
            results[name] = analysis_results
            
            # Print summary
            summary = analyzer.get_complexity_summary(analysis_results)
            print(f"  Overall Complexity: {summary.get('overall_complexity', 'Unknown')}")
            print(f"  Synergy Level: {summary.get('synergy_level', 'Unknown')}")
            print(f"  Temporal Complexity: {summary.get('temporal_complexity', 'Unknown')}")
            print(f"  Modularity Level: {summary.get('modularity_level', 'Unknown')}")
            print()
        
        # Print detailed results
        print_detailed_results(results)
        
        # Create visualizations
        print("Creating visualizations...")
        visualize_results(systems, results)
        
        print("Analysis complete! Results saved as 'complexity_analysis_results.png'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
