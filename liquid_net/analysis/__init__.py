"""
Analysis module for Twistor-inspired Liquid Neural Network.

Provides tools for analyzing the dynamical system:
- Fixed point analysis
- Jacobian eigenvalue computation
- Phase space visualization
- Stability analysis
- Bifurcation diagrams
"""

from .dynamics import DynamicsAnalyzer, plot_bifurcation_diagram

__all__ = [
    'DynamicsAnalyzer',
    'plot_bifurcation_diagram',
]
