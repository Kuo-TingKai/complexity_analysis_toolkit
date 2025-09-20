"""
Dynamics Analysis Module

This module implements dynamical complexity measures:
- Multiscale entropy: entropy across different temporal scales
- Fractal scaling: self-similarity and scaling properties
- Criticality: measures of system criticality and phase transitions
"""

import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import warnings
from typing import List, Tuple, Dict, Optional, Union


class DynamicsAnalyzer:
    """
    Analyzer for dynamical complexity measures
    """
    
    def __init__(self, tolerance: float = 0.2, max_scale: int = 20):
        """
        Initialize the dynamics analyzer
        
        Args:
            tolerance: Tolerance for sample entropy calculation (fraction of std)
            max_scale: Maximum scale for multiscale entropy
        """
        self.tolerance = tolerance
        self.max_scale = max_scale
    
    def _coarse_grain(self, data: np.ndarray, scale: int) -> np.ndarray:
        """
        Coarse-grain time series by averaging over scale points
        
        Args:
            data: Input time series
            scale: Scale factor for coarse-graining
            
        Returns:
            Coarse-grained time series
        """
        if scale == 1:
            return data
        
        # Pad data to ensure integer division
        n = len(data)
        pad_length = (scale - n % scale) % scale
        if pad_length > 0:
            padded_data = np.pad(data, (0, pad_length), mode='edge')
        else:
            padded_data = data
        
        # Reshape and average
        reshaped = padded_data.reshape(-1, scale)
        return np.mean(reshaped, axis=1)
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, tolerance: Optional[float] = None) -> float:
        """
        Calculate sample entropy of time series
        
        Args:
            data: Input time series
            m: Embedding dimension
            tolerance: Tolerance for matching (if None, uses self.tolerance * std)
            
        Returns:
            Sample entropy value
        """
        if tolerance is None:
            tolerance = self.tolerance * np.std(data)
        
        n = len(data)
        if n < m + 1:
            return 0.0
        
        # Ensure minimum data length for reliable calculation
        if n < 50:
            return np.nan
        
        # Ensure tolerance is not too small (avoid division by zero issues)
        if tolerance <= 0:
            tolerance = 0.001
        
        # Create template vectors
        templates = np.array([data[i:i+m] for i in range(n-m+1)])
        
        # Count matches for m and m+1 dimensional vectors
        matches_m = 0
        matches_m1 = 0
        valid_comparisons = 0
        
        for i in range(len(templates)):
            # Find matches for m-dimensional vectors
            distances_m = np.abs(templates - templates[i]).max(axis=1)
            matches_m += np.sum(distances_m <= tolerance) - 1  # Exclude self-match
            
            # Find matches for m+1 dimensional vectors
            if i < len(templates) - 1 and i + m < len(data):
                template_m1 = np.append(templates[i], data[i+m])
                # Create m+1 dimensional templates for comparison
                if i + m < len(data):
                    templates_m1 = []
                    for k in range(len(templates)):
                        if k + m < len(data):
                            templates_m1.append(np.append(templates[k], data[k+m]))
                    
                    if templates_m1:
                        templates_m1 = np.array(templates_m1)
                        distances_m1 = np.abs(templates_m1 - template_m1).max(axis=1)
                        matches_m1 += np.sum(distances_m1 <= tolerance) - 1
                        valid_comparisons += 1
        
        # Calculate sample entropy with better numerical stability
        if matches_m == 0 or valid_comparisons == 0:
            return 0.0
        
        # Normalize by number of valid comparisons
        avg_matches_m = matches_m / len(templates)
        avg_matches_m1 = matches_m1 / valid_comparisons
        
        # Avoid log(0) and ensure numerical stability
        if avg_matches_m1 <= 0 or avg_matches_m <= 0:
            return 0.0
        
        ratio = avg_matches_m1 / avg_matches_m
        if ratio <= 0 or ratio >= 1:
            return 0.0
        
        try:
            return -np.log(ratio)
        except (ValueError, OverflowError):
            return np.nan
    
    def multiscale_entropy(self, data: np.ndarray, scales: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Calculate multiscale entropy across different temporal scales
        
        Args:
            data: Input time series
            scales: List of scales to analyze (if None, uses 1 to max_scale)
            
        Returns:
            Dictionary mapping scales to sample entropy values
        """
        if scales is None:
            scales = list(range(1, self.max_scale + 1))
        
        mse_results = {}
        
        for scale in scales:
            try:
                # Coarse-grain the data
                coarse_data = self._coarse_grain(data, scale)
                
                # Check if coarse-grained data is long enough
                if len(coarse_data) < 50:
                    mse_results[scale] = np.nan
                    continue
                
                # Calculate sample entropy
                entropy_val = self._sample_entropy(coarse_data)
                mse_results[scale] = entropy_val
                
            except Exception as e:
                warnings.warn(f"Failed to calculate MSE for scale {scale}: {e}")
                mse_results[scale] = np.nan
        
        return mse_results
    
    def _detrended_fluctuation_analysis(self, data: np.ndarray, scales: List[int]) -> Tuple[np.ndarray, float]:
        """
        Perform detrended fluctuation analysis
        
        Args:
            data: Input time series
            scales: List of scales for analysis
            
        Returns:
            Tuple of (fluctuations, scaling exponent)
        """
        # Integrate the time series
        y = np.cumsum(data - np.mean(data))
        
        fluctuations = []
        
        for scale in scales:
            # Divide into segments
            n_segments = len(y) // scale
            if n_segments < 2:
                continue
                
            # Calculate local trend and fluctuation for each segment
            segment_fluctuations = []
            
            for i in range(n_segments):
                start = i * scale
                end = start + scale
                segment = y[start:end]
                
                # Detrend (remove linear trend)
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                detrended = segment - trend
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean(detrended**2))
                segment_fluctuations.append(fluctuation)
            
            # Average fluctuation for this scale
            avg_fluctuation = np.mean(segment_fluctuations)
            fluctuations.append(avg_fluctuation)
        
        # Fit power law
        valid_scales = [s for s, f in zip(scales, fluctuations) if f > 0]
        valid_fluctuations = [f for f in fluctuations if f > 0]
        
        if len(valid_scales) < 2:
            return np.array(fluctuations), np.nan
        
        # Linear regression in log-log space
        log_scales = np.log(valid_scales)
        log_fluctuations = np.log(valid_fluctuations)
        
        slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)
        
        return np.array(fluctuations), slope
    
    def fractal_scaling(self, data: np.ndarray, scales: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Calculate fractal scaling properties
        
        Args:
            data: Input time series
            scales: List of scales for analysis (if None, uses logarithmic spacing)
            
        Returns:
            Dictionary containing fractal scaling measures
        """
        if scales is None:
            # Use logarithmic spacing of scales
            scales = [int(2**i) for i in range(2, int(np.log2(len(data)/4)))]
        
        results = {}
        
        try:
            # Detrended Fluctuation Analysis
            dfa_fluctuations, dfa_alpha = self._detrended_fluctuation_analysis(data, scales)
            results['dfa_alpha'] = dfa_alpha
            
            # Hurst exponent (alternative calculation)
            if not np.isnan(dfa_alpha):
                results['hurst_exponent'] = dfa_alpha
            
        except Exception as e:
            warnings.warn(f"Failed to calculate DFA: {e}")
            results['dfa_alpha'] = np.nan
            results['hurst_exponent'] = np.nan
        
        try:
            # Power spectral density analysis
            fft_data = fft(data - np.mean(data))
            freqs = fftfreq(len(data))
            psd = np.abs(fft_data)**2
            
            # Keep only positive frequencies
            pos_freqs = freqs[1:len(freqs)//2]
            pos_psd = psd[1:len(psd)//2]
            
            # Fit power law in log-log space
            log_freqs = np.log(pos_freqs[pos_freqs > 0])
            log_psd = np.log(pos_psd[pos_freqs > 0])
            
            if len(log_freqs) > 1:
                slope, _, _, _, _ = stats.linregress(log_freqs, log_psd)
                results['spectral_slope'] = -slope  # Negative because PSD ∝ f^(-β)
            else:
                results['spectral_slope'] = np.nan
                
        except Exception as e:
            warnings.warn(f"Failed to calculate spectral slope: {e}")
            results['spectral_slope'] = np.nan
        
        return results
    
    def _avalanche_analysis(self, data: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Perform avalanche analysis for criticality detection
        
        Args:
            data: Input time series (should be binary or thresholded)
            threshold: Threshold for binarization (if None, uses median)
            
        Returns:
            Dictionary containing avalanche statistics
        """
        if threshold is None:
            threshold = np.median(data)
        
        # Binarize the data
        binary_data = (data > threshold).astype(int)
        
        # Find avalanches (continuous sequences of 1s)
        avalanches = []
        current_avalanche = 0
        
        for val in binary_data:
            if val == 1:
                current_avalanche += 1
            else:
                if current_avalanche > 0:
                    avalanches.append(current_avalanche)
                    current_avalanche = 0
        
        # Don't forget the last avalanche if it ends with 1
        if current_avalanche > 0:
            avalanches.append(current_avalanche)
        
        avalanches = np.array(avalanches)
        
        if len(avalanches) == 0:
            return {'avalanche_sizes': np.array([]), 'power_law_exponent': np.nan, 'criticality_index': np.nan}
        
        # Calculate power law exponent
        unique_sizes, counts = np.unique(avalanches, return_counts=True)
        
        if len(unique_sizes) < 3:
            return {'avalanche_sizes': avalanches, 'power_law_exponent': np.nan, 'criticality_index': np.nan}
        
        # Fit power law (P(s) ∝ s^(-τ))
        log_sizes = np.log(unique_sizes[unique_sizes > 0])
        log_counts = np.log(counts[unique_sizes > 0])
        
        if len(log_sizes) > 1:
            slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)
            power_law_exponent = -slope
        else:
            power_law_exponent = np.nan
        
        # Criticality index (how close to critical value of 1.5)
        criticality_index = 1.0 - abs(power_law_exponent - 1.5) / 1.5 if not np.isnan(power_law_exponent) else np.nan
        
        return {
            'avalanche_sizes': avalanches,
            'power_law_exponent': power_law_exponent,
            'criticality_index': criticality_index
        }
    
    def criticality(self, data: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate criticality measures
        
        Args:
            data: Input time series
            threshold: Threshold for avalanche analysis (if None, uses median)
            
        Returns:
            Dictionary containing criticality measures
        """
        results = {}
        
        try:
            # Avalanche analysis
            avalanche_results = self._avalanche_analysis(data, threshold)
            results['power_law_exponent'] = avalanche_results['power_law_exponent']
            results['criticality_index'] = avalanche_results['criticality_index']
            
        except Exception as e:
            warnings.warn(f"Failed to perform avalanche analysis: {e}")
            results['power_law_exponent'] = np.nan
            results['criticality_index'] = np.nan
        
        try:
            # Coefficient of variation (CV) - indicator of criticality
            cv = np.std(data) / np.mean(data) if np.mean(data) != 0 else np.nan
            results['coefficient_of_variation'] = cv
            
        except Exception as e:
            warnings.warn(f"Failed to calculate coefficient of variation: {e}")
            results['coefficient_of_variation'] = np.nan
        
        try:
            # Autocorrelation at lag 1
            if len(data) > 1:
                autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
                results['autocorrelation_lag1'] = autocorr if not np.isnan(autocorr) else 0
            else:
                results['autocorrelation_lag1'] = 0
                
        except Exception as e:
            warnings.warn(f"Failed to calculate autocorrelation: {e}")
            results['autocorrelation_lag1'] = 0
        
        return results
    
    def analyze(self, data: np.ndarray, scales: Optional[List[int]] = None) -> Dict[str, Union[float, Dict]]:
        """
        Perform comprehensive dynamical analysis
        
        Args:
            data: Input time series
            scales: List of scales for analysis (if None, uses default)
            
        Returns:
            Dictionary containing all dynamical measures
        """
        results = {}
        
        # Multiscale entropy
        try:
            mse_results = self.multiscale_entropy(data, scales)
            results['multiscale_entropy'] = mse_results
            
            # Summary statistics for MSE
            valid_entropies = [v for v in mse_results.values() if not np.isnan(v)]
            if valid_entropies:
                results['mse_mean'] = np.mean(valid_entropies)
                results['mse_std'] = np.std(valid_entropies)
                results['mse_slope'] = np.polyfit(list(mse_results.keys()), list(mse_results.values()), 1)[0]
            else:
                results['mse_mean'] = np.nan
                results['mse_std'] = np.nan
                results['mse_slope'] = np.nan
                
        except Exception as e:
            warnings.warn(f"Failed to calculate multiscale entropy: {e}")
            results['multiscale_entropy'] = {}
            results['mse_mean'] = np.nan
            results['mse_std'] = np.nan
            results['mse_slope'] = np.nan
        
        # Fractal scaling
        try:
            fractal_results = self.fractal_scaling(data, scales)
            results.update(fractal_results)
        except Exception as e:
            warnings.warn(f"Failed to calculate fractal scaling: {e}")
        
        # Criticality
        try:
            criticality_results = self.criticality(data)
            results.update(criticality_results)
        except Exception as e:
            warnings.warn(f"Failed to calculate criticality: {e}")
        
        return results
