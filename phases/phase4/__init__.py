"""
Phase 4: Thermodynamic Learning (TL)

Learning through energy minimization, not backpropagation.
System settles to minimum energy states like physical systems.

Key components:
- ThermodynamicLearner: Main learning engine
- Temperature: Controls exploration vs exploitation
- EnergyFunctions: Defines energy landscapes
- OnlineLearner: Continuous learning from stream

Usage:
    from phases.phase4 import ThermodynamicLearner, Temperature
    
    learner = ThermodynamicLearner(field, temperature=1.0)
    learner.settle(input_wave, steps=100)  # Energy minimization
"""

try:
    from .thermodynamic import (
        ThermodynamicLearner,
        settle_to_equilibrium,
        compute_energy,
    )
    from .temperature import (
        Temperature,
        TemperatureSchedule,
        anneal,
        get_temperature,
    )
    from .energy_functions import (
        EnergyFunction,
        hamiltonian_energy,
        boltzmann_distribution,
        free_energy,
    )
    from .online_learner import (
        OnlineLearner,
        StreamBuffer,
        continuous_update,
    )
except ImportError:
    # Allow partial imports for standalone execution
    pass

__all__ = [
    # Thermodynamic learner
    'ThermodynamicLearner',
    'settle_to_equilibrium',
    'compute_energy',
    
    # Temperature control
    'Temperature',
    'TemperatureSchedule',
    'anneal',
    'get_temperature',
    
    # Energy functions
    'EnergyFunction',
    'hamiltonian_energy',
    'boltzmann_distribution',
    'free_energy',
    
    # Online learning
    'OnlineLearner',
    'StreamBuffer',
    'continuous_update',
]
