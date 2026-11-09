# Dynamic Multiphase Modeling of Biomass Pyrolysis & Secondary Tar Cracking in a Fixed-Bed Reactor

## Project Overview

This project implements a rigorous computational model for biomass pyrolysis in a fixed-bed reactor, combining:
- Multiphase transport phenomena (solid-gas interphase heat transfer)
- Non-linear reaction kinetics (primary and secondary reactions)
- Solid-fluid heat transfer
- Numerical PDE solving via Method of Lines

## Mathematical Formulation

### Solid Phase
- **Devolatilization Kinetics**: Multi-reaction parallel scheme (Cellulose, Hemicellulose, Lignin → Char + Tar + Gas)
- **Energy Balance**: Includes intra-particle heat transfer and interphase convective heat transfer

### Gas Phase
- **Species Conservation**: Advection, axial dispersion, and secondary tar cracking
- **Energy Balance**: Gas-phase energy with interphase heat transfer
- **Fluid Dynamics**: Ergun equation for pressure drop profiles

### Key Equations
- Biomass Conservation: ∂ρ_s/∂t = -(k₁ + k₂ + k₃)ρ_s where k_i = A_i exp(-E_i/RT_s)
- Gas Species: ε ∂C_i/∂t + ∂(u_g C_i)/∂z = D_ax ∂²C_i/∂z² + R_gen,i - r_cracking,i
- Ergun Equation: -dp/dz = 150μ(1-ε)²/(φ²d_p²ε³) u_g + 1.75(1-ε)/(φd_pε³) ρ_g u_g²

## Implementation Plan

### Phase 1: Foundations & Kinetics Setup (Weeks 1-2)
- Define kinetic triplets from literature
- Implement single-particle TGA validation script

### Phase 2: 1D Spatial Discretization & Numerical Solver (Weeks 3-5)
- Method of Lines (MOL) in Python using scipy.integrate.solve_ivp
- Solid and gas energy balances with interphase heat transfer
- Dynamic Ergun equation integration

### Phase 3: Secondary Cracking & Yield Optimization (Weeks 6-7)
- Add gas-phase secondary tar cracking kinetics
- Parametric sensitivity analysis

### Phase 4: Validation & Visualization Dashboard (Weeks 8)
- Benchmark against experimental data
- Interactive dashboard for axial temperature profiles

## Tech Stack

- **Language**: Python 3.10+
- **Numerical Core**: NumPy, SciPy (solve_ivp with Radau/BDF)
- **Visualization**: Matplotlib/Seaborn
- **Optional UI**: Streamlit

## Kinetic Parameters

Standard Arrhenius parameters from literature (Ranzi et al., Di Blasi, Broido-Shafizadeh):

| Component/Pathway | Pre-exponential Factor (A) | Activation Energy (Ea) | Heat of Reaction (ΔH) |
|---|---|---|---|
| Hemicellulose → Volatiles + Char | 2.10×10¹⁶ s⁻¹ | 100.0 kJ/mol | -80 kJ/kg |
| Cellulose → Tar (Levoglucosan) | 3.28×10¹⁴ s⁻¹ | 196.0 kJ/mol | +150 kJ/kg |
| Cellulose → Char + Gases | 1.30×10¹⁰ s⁻¹ | 150.0 kJ/mol | +150 kJ/kg |
| Lignin (Pathway 1) → Phenolic Tar | 9.60×10⁸ s⁻¹ | 112.5 kJ/mol | ~0 kJ/kg |
| Lignin (Pathway 2) → Char + Gas | 1.50×10⁹ s⁻¹ | 120.0 kJ/mol | ~0 kJ/kg |
| Secondary Tar Cracking | 4.28×10⁵ s⁻¹ | 108.0 kJ/mol | N/A |

## Getting Started

1. Install dependencies: `pip install numpy scipy matplotlib streamlit`
2. Run the main solver: `python src/pyrolysis_model.py`
3. Launch dashboard: `streamlit run src/dashboard.py`"# biomass-pyrolisis-fixed-bed-reactor" 
