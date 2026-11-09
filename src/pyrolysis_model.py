"""
Dynamic Multiphase Modeling of Biomass Pyrolysis & Secondary Tar Cracking
in a Fixed-Bed Reactor using Method of Lines (MOL)
"""

import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib.pyplot as plt


# ============================================================
# KINETIC PARAMETERS (from literature: Ranzi et al., Di Blasi, Broido-Shafizadeh)
# ============================================================

# Gas constant
R = 8.314  # J/(mol·K)

# === PRIMARY PYROLYSIS KINETICS (Biomass Devolatilization) ===

# Hemicellulose -> Volatiles + Char
A_hemicell = 2.10e16    # 1/s
E_hemicell = 100.0e3    # J/mol
dH_hemicell = -80.0     # kJ/kg (mildly exothermic)

# Cellulose -> Tar (Levoglucosan)
A_cell_tar = 3.28e14    # 1/s
E_cell_tar = 196.0e3    # J/mol
dH_cell_tar = 150.0     # kJ/kg (endothermic)

# Cellulose -> Char + Gases
A_cell_char = 1.30e10   # 1/s
E_cell_char = 150.0e3   # J/mol
dH_cell_char = 150.0    # kJ/kg (endothermic)

# Lignin (Pathway 1) -> Phenolic Tar
A_lignin1 = 9.60e8      # 1/s
E_lignin1 = 112.5e3     # J/mol
dH_lignin1 = 0.0        # kJ/kg (neutral)

# Lignin (Pathway 2) -> Char + Gas
A_lignin2 = 1.50e9      # 1/s
E_lignin2 = 120.0e3     # J/mol
dH_lignin2 = 0.0        # kJ/kg (neutral)

# === SECONDARY GAS-PHASE TAR CRACKING KINETICS ===
A_cracking = 4.28e5     # 1/s
E_cracking = 108.0e3    # J/mol


# ============================================================
# KINETIC RATE CONSTANTS
# ============================================================

def calculate_kinetics(T_s):
    """
    Calculate reaction rate constants based on solid temperature T_s.
    
    Parameters
    ----------
    T_s : float or np.array
        Solid phase temperature [K] or array of temperatures
        
    Returns
    -------
    dict of rate constants
    """
    # Handle both scalar and array inputs
    T_s_arr = np.asarray(T_s, dtype=float)
    
    k_tar = A_cell_tar * np.exp(-E_cell_tar / (R * T_s_arr))
    k_char = A_cell_char * np.exp(-E_cell_char / (R * T_s_arr))
    k_hemicell = A_hemicell * np.exp(-E_hemicell / (R * T_s_arr))
    k_lignin1 = A_lignin1 * np.exp(-E_lignin1 / (R * T_s_arr))
    k_lignin2 = A_lignin2 * np.exp(-E_lignin2 / (R * T_s_arr))
    k_cracking = A_cracking * np.exp(-E_cracking / (R * T_s_arr))
    
    # If input was scalar, return scalars; if array, return arrays
    if np.isscalar(T_s) or T_s.ndim == 0:
        return {
            'k_tar': float(k_tar),
            'k_char': float(k_char),
            'k_hemicell': float(k_hemicell),
            'k_lignin1': float(k_lignin1),
            'k_lignin2': float(k_lignin2),
            'k_cracking': float(k_cracking)
        }
    else:
        return {
            'k_tar': k_tar,
            'k_char': k_char,
            'k_hemicell': k_hemicell,
            'k_lignin1': k_lignin1,
            'k_lignin2': k_lignin2,
            'k_cracking': k_cracking
        }


# ============================================================
# GOVERNING EQUATIONS - METHOD OF LINES (MOL)
# ============================================================

def pyrolysis_odes(t, y, params):
    """
    System of ODEs for the 1D fixed-bed pyrolysis reactor.
    
    State vector y = [ρ_s, T_s, C_tar, C_gases, T_g, p] along spatial grid
    """
    N = params['N']
    dz = params['dz']
    
    # Unpack state variables
    rho_s = y[0:N]           # Solid biomass density [kg/m³]
    T_s = y[N:2*N]           # Solid temperature [K]
    C_tar = y[2*N:3*N]       # Tar concentration [kmol/m³]
    C_gases = y[3*N:4*N]     # Gas species concentration [kmol/m³]
    T_g = y[4*N:5*N]         # Gas temperature [K]
    p = y[5*N:6*N]           # Pressure [Pa]
    
    # Extract parameters (all scalars in default_params)
    epsilon = params['epsilon']
    rho_g = params['rho_g']
    Cp_g = params['Cp_g']
    Cp_s = params['Cp_s']
    k_eff = params['k_eff']
    k_g_eff = params['k_g_eff']
    h_v_a_v = params['h_v_a_v']
    u_g = params['u_g']
    D_ax = params['D_ax']
    d_p = params['d_p']
    phi = params['phi']
    mu = params['mu']
    
    # ============================================================
    # SOLID PHASE: Calculate kinetics
    # ============================================================
    
    kinetics = calculate_kinetics(T_s)
    
    # Get rate constants - handle scalar vs array output
    kt = kinetics['k_tar']
    kc = kinetics['k_char']
    kh = kinetics['k_hemicell']
    kl1 = kinetics['k_lignin1']
    kl2 = kinetics['k_lignin2']
    kcrack = kinetics['k_cracking']
    
    # Ensure rate constants are arrays matching T_s shape for element-wise operations
    T_s_arr = np.asarray(T_s, dtype=float)
    if np.isscalar(kt) or kt.ndim == 0 or np.isscalar(kcrack):
        # Single temperature point - broadcast to all N points
        kt_arr = np.full(N, float(kt))
        kc_arr = np.full(N, float(kc))
        kh_arr = np.full(N, float(kh))
        kl1_arr = np.full(N, float(kl1))
        kl2_arr = np.full(N, float(kl2))
        kcrack_arr = np.full(N, float(kcrack))
    else:
        kt_arr = np.asarray(kt, dtype=float)
        kc_arr = np.asarray(kc, dtype=float)
        kh_arr = np.asarray(kh, dtype=float)
        kl1_arr = np.asarray(kl1, dtype=float)
        kl2_arr = np.asarray(kl2, dtype=float)
        kcrack_arr = np.asarray(kcrack, dtype=float)
    
    # --- Biomass Conservation: dρ_s/dt = -(k1+k2+k3)ρ_s ---
    total_k = kh_arr + kt_arr + kc_arr + kl1_arr + kl2_arr
    
    d_rho_s_dt = -total_k * rho_s
    
    # --- Solid Energy Balance ---
    # (1-ε)ρ_s Cp_s ∂T_s/∂t = ∂/∂z(k_eff ∂T_s/∂z) - h_v a_v (T_s - T_g) + Σ(-ΔH_rxn,i) R_i
    
    dT_s_dt = np.zeros(N)
    for i in range(N):
        # Conduction term (finite difference with boundary handling)
        if i == 0:
            # Left boundary: Neumann (insulated, dT/dz = 0)
            d2T_s_dz2 = (T_s[i+1] - 2*T_s[i] + T_s[i]) / dz**2
        elif i == N-1:
            # Right boundary: insulated
            d2T_s_dz2 = (T_s[i] - 2*T_s[i] + T_s[i-1]) / dz**2
        else:
            # Interior: central difference
            d2T_s_dz2 = (T_s[i+1] - 2*T_s[i] + T_s[i-1]) / dz**2
        
        # Interphase heat transfer
        heat_transfer = -h_v_a_v * (T_s[i] - T_g[i])
        
        # Reaction heat source
        # R_i = k_i * ρ_s at position i - use [i] indexing for arrays
        R_h = kh_arr[i] * rho_s[i]
        R_ct = kt_arr[i] * rho_s[i]
        R_cc = kc_arr[i] * rho_s[i]
        R_l1 = kl1_arr[i] * rho_s[i]
        R_l2 = kl2_arr[i] * rho_s[i]
        
        heat_reaction = (
            -dH_hemicell/1000 * R_h +
            -dH_cell_tar/1000 * R_ct +
            -dH_cell_char/1000 * R_cc +
            -dH_lignin1/1000 * R_l1 +
            -dH_lignin2/1000 * R_l2
        )
        
        dT_s_dt[i] = (k_eff * d2T_s_dz2 + heat_transfer + heat_reaction) / ((1 - epsilon) * rho_s[i] * Cp_s)
    
    # ============================================================
    # GAS PHASE: Tar Species Conservation
    # ε ∂C_tar/∂t + ∂(u_g C_tar)/∂z = D_ax ∂²C_tar/∂z² + R_gen,tar - r_cracking,tar
    # ============================================================
    
    dC_tar_dt = np.zeros(N)
    for i in range(N):
        # Advection term
        if i == 0:
            dC_dz_adv = (C_tar[i+1] - C_tar[i]) / dz if N > 1 else 0.0
        elif i == N-1:
            dC_dz_adv = (C_tar[i] - C_tar[i-1]) / dz
        else:
            dC_dz_adv = (C_tar[i+1] - C_tar[i-1]) / (2 * dz)
        
        adv_term = u_g * dC_dz_adv
        
        # Dispersion term
        if i == 0 or i == N-1:
            d2C_dz2 = 0.0
        else:
            d2C_dz2 = (C_tar[i+1] - 2*C_tar[i] + C_tar[i-1]) / dz**2
        disp_term = D_ax * d2C_dz2
        
        # Generation from primary pyrolysis (tar from cellulose)
        # Use [i] indexing: kt_arr[i] * rho_s[i]
        R_gen_tar = kt_arr[i] * rho_s[i]
        
        # Secondary cracking loss - use [i] indexing for array rate constant
        r_cracking = kcrack_arr[i] * C_tar[i]
        
        dC_tar_dt[i] = (-epsilon * adv_term + disp_term + R_gen_tar - r_cracking) / epsilon
    
    # ============================================================
    # GAS PHASE: Total Gases Conservation
    # ============================================================
    
    dC_gases_dt = np.zeros(N)
    for i in range(N):
        # Generation from all primary reactions at position i - use [i] indexing
        R_gen_gases = (
            kh_arr[i] * rho_s[i] +
            kc_arr[i] * rho_s[i] +
            kl1_arr[i] * rho_s[i] +
            kl2_arr[i] * rho_s[i]
        )
        # Cracking produces additional gases - use [i] indexing
        r_cracking = kcrack_arr[i] * C_tar[i]
        dC_gases_dt[i] = (R_gen_gases + r_cracking)
    
    # ============================================================
    # GAS PHASE: Energy Balance
    # ερ_g Cp_g (∂T_g/∂t + u_g ∂T_g/∂z) = k_g,eff ∂²T_g/∂z² + h_v a_v (T_s - T_g)
    # ============================================================
    
    dT_g_dt = np.zeros(N)
    for i in range(N):
        # Advection term
        if i == 0:
            dTg_dz = (T_g[i+1] - T_g[i]) / dz if N > 1 else 0.0
        elif i == N-1:
            dTg_dz = (T_g[i] - T_g[i-1]) / dz
        else:
            dTg_dz = (T_g[i+1] - T_g[i-1]) / (2 * dz)
        
        adv_term = u_g * dTg_dz
        
        # Dispersion term
        if i == 0 or i == N-1:
            d2Tg_dz2 = 0.0
        else:
            d2Tg_dz2 = (T_g[i+1] - 2*T_g[i] + T_g[i-1]) / dz**2
        disp_term = k_g_eff * d2Tg_dz2
        
        # Interphase heat transfer
        heat_transfer = h_v_a_v * (T_s[i] - T_g[i])
        
        dT_g_dt[i] = (disp_term + heat_transfer - epsilon * rho_g * Cp_g * adv_term) / (epsilon * rho_g * Cp_g)
    
    # ============================================================
    # FLUID DYNAMICS: Ergun Equation for Pressure Drop
    # -dp/dz = 150μ(1-ε)²/(φ²d_p²ε³) u_g + 1.75(1-ε)/(φd_pε³) ρ_g u_g²
    # ============================================================
    
    dp_dz = np.zeros(N)
    for i in range(N):
        term1 = 150 * mu * (1 - epsilon)**2 / (phi**2 * d_p**2 * epsilon**3) * u_g
        term2 = 1.75 * (1 - epsilon) / (phi * d_p * epsilon**3) * rho_g * u_g**2
        dp_dz[i] = -(term1 + term2)
    
    # ============================================================
    # ASSEMBLE DERIVATIVE VECTOR
    # ============================================================
    
    dydt = np.zeros(6 * N)
    dydt[0:N] = d_rho_s_dt
    dydt[N:2*N] = dT_s_dt
    dydt[2*N:3*N] = dC_tar_dt
    dydt[3*N:4*N] = dC_gases_dt
    dydt[4*N:5*N] = dT_g_dt
    dydt[5*N:6*N] = dp_dz
    
    return dydt


# ============================================================
# SOLVER SETUP AND MAIN SIMULATION
# ============================================================

def run_simulation(params=None):
    """
    Run the biomass pyrolysis simulation.
    
    Parameters
    ----------
    params : dict, optional
        Physical and geometric parameters. If None, uses default values.
        
    Returns
    -------
    sol : OdeResult
        Solution object from solve_ivp
    t : np.array
        Time points
    params : dict
        Final parameters
    """
    if params is None:
        params = default_params()
    
    N = params['N']
    L = params['L']
    dz = L / (N - 1)
    params['dz'] = dz
    
    # Initial conditions
    y0 = np.zeros(6 * N)
    y0[0:N] = params['rho_s0']       # Initial biomass density [kg/m³]
    y0[N:2*N] = params['T_s0']       # Initial solid temperature [K]
    y0[2*N:3*N] = 0.0                # No tar initially [kmol/m³]
    y0[3*N:4*N] = params['C_gases0'] # Initial gas species concentration [kmol/m³]
    y0[4*N:5*N] = params['T_g0']     # Initial gas temperature [K]
    y0[5*N:6*N] = params['p0']       # Initial pressure [Pa]
    
    t_span = (0, params['t_max'])
    
    solver_options = {
        'method': 'Radau',
        'rtol': 1e-6,
        'atol': 1e-8,
        'max_step': params.get('max_step', 1.0)
    }
    
    sol = solve_ivp(
        pyrolysis_odes,
        t_span,
        y0,
        method=solver_options['method'],
        rtol=solver_options['rtol'],
        atol=solver_options['atol'],
        max_step=solver_options['max_step'],
        args=(params,),
        vectorized=False
    )
    
    return sol, params


def default_params():
    """Default parameters for a typical biomass pyrolysis fixed-bed reactor."""
    return {
        'N': 50,
        'L': 1.0,
        'dz': 0.02,
        'rho_s0': 500.0,
        'T_s0': 300.0,
        'Cp_s': 1500.0,
        'rho_s_bulk': 250.0,
        'k_eff': 0.5,
        'epsilon': 0.4,
        'rho_g': 1.2,
        'Cp_g': 1000.0,
        'k_g_eff': 0.03,
        'u_g': 0.1,
        'D_ax': 0.01,
        'd_p': 0.005,
        'phi': 0.85,
        'mu': 2.5e-5,
        'h_v_a_v': 50.0,
        'C_gases0': 0.05,
        'T_g0': 300.0,
        'p0': 101325.0,
        't_max': 600.0
    }


# ============================================================
# POST-PROCESSING AND VISUALIZATION
# ============================================================

def extract_results(sol, params):
    """
    Extract and organize simulation results for visualization.
    
    Uses cumulative_trapezoid for yield integration (compatible with NumPy 2.x).
    """
    N = params['N']
    t = sol.t
    y = sol.y
    z = np.linspace(0, params['L'], N)
    
    results = {
        'time': t,
        'z': z,
        'N': N,
        'L': params['L'],
        'rho_s': y[0:N, :].T,
        'T_s': y[N:2*N, :].T,
        'C_tar': y[2*N:3*N, :].T,
        'C_gases': y[3*N:4*N, :].T,
        'T_g': y[4*N:5*N, :].T,
        'p': y[5*N:6*N, :].T,
    }
    
    # Compute yields using cumulative_trapezoid along spatial axis (axis=1)
    # For each time step, integrate C_tar and C_gases over z
    # cumulative_trapezoid returns array of same shape, we take the last value
    tar_yields = cumulative_trapezoid(results['C_tar'], results['z'], axis=1)
    gases_yields = cumulative_trapezoid(results['C_gases'], results['z'], axis=1)
    
    results['tar_yield'] = tar_yields[:, -1]  # Final value at each time step -> take last spatial point
    results['gases_yield'] = gases_yields[:, -1]
    
    idx_end = len(t) - 1
    results['T_s_end'] = results['T_s'][idx_end, :]
    results['T_g_end'] = results['T_g'][idx_end, :]
    results['C_tar_end'] = results['C_tar'][idx_end, :]
    
    return results


def plot_results(results, save_path=None):
    """
    Plot simulation results: axial temperature profiles and yield evolution.
    """
    t = results['time']
    z = results['z']
    T_s = results['T_s']
    T_g = results['T_g']
    C_tar = results['C_tar']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Biomass Pyrolysis Fixed-Bed Reactor Simulation', fontsize=14)
    
    idx_start = 0
    idx_mid = len(t) // 2
    idx_end = len(t) - 1
    
    # Solid temperature profiles
    ax1 = axes[0, 0]
    ax1.plot(z, T_s[idx_start, :], 'b-', label=f't={t[idx_start]:.1f}s', alpha=0.7)
    ax1.plot(z, T_s[idx_mid, :], 'b-', label=f't={t[idx_mid]:.1f}s', alpha=0.7)
    ax1.plot(z, T_s[idx_end, :], 'b-', label=f't={t[idx_end]:.1f}s', linewidth=2)
    ax1.set_xlabel('Reactor length z [m]')
    ax1.set_ylabel('Solid temperature T_s [K]')
    ax1.set_title('Solid Phase Temperature Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gas temperature profiles
    ax2 = axes[0, 1]
    ax2.plot(z, T_g[idx_start, :], 'r--', label=f't={t[idx_start]:.1f}s', alpha=0.7)
    ax2.plot(z, T_g[idx_mid, :], 'r--', label=f't={t[idx_mid]:.1f}s', alpha=0.7)
    ax2.plot(z, T_g[idx_end, :], 'r--', label=f't={t[idx_end]:.1f}s', linewidth=2)
    ax2.set_xlabel('Reactor length z [m]')
    ax2.set_ylabel('Gas temperature T_g [K]')
    ax2.set_title('Gas Phase Temperature Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Tar concentration profiles
    ax3 = axes[1, 0]
    ax3.plot(z, C_tar[idx_start, :], 'g-.', label=f't={t[idx_start]:.1f}s', alpha=0.7)
    ax3.plot(z, C_tar[idx_mid, :], 'g-.', label=f't={t[idx_mid]:.1f}s', alpha=0.7)
    ax3.plot(z, C_tar[idx_end, :], 'g-', label=f't={t[idx_end]:.1f}s', linewidth=2)
    ax3.set_xlabel('Reactor length z [m]')
    ax3.set_ylabel('Tar concentration C_tar [kmol/m³]')
    ax3.set_title('Tar Concentration Profiles (with Secondary Cracking)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Yield evolution
    ax4 = axes[1, 1]
    ax4.plot(t/60, results['tar_yield'], 'mo-', label='Tar yield', linewidth=2)
    ax4.plot(t/60, results['gases_yield'], 'co-', label='Gases yield', linewidth=2)
    ax4.set_xlabel('Time [minutes]')
    ax4.set_ylabel('Yield [kmol]')
    ax4.set_title('Integrated Yields Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Biomass Pyrolysis Fixed-Bed Reactor Simulation")
    print("=" * 60)
    print()
    print("Running simulation with default parameters...")
    print()
    
    sol, params = run_simulation()
    results = extract_results(sol, params)
    
    print("Generating plots...")
    fig = plot_results(results)
    
    print()
    print("Simulation complete!")
    print(f"Time points: {len(sol.t)}")
    print(f"Spatial grid points: {params['N']}")
    print(f"Final solid temp max: {np.max(results['T_s'][-1, :]):.1f} K")
    print(f"Final gas temp max: {np.max(results['T_g'][-1, :]):.1f} K")
    print(f"Final tar yield: {results['tar_yield'][-1]:.4f} kmol")
    print(f"Final gases yield: {results['gases_yield'][-1]:.4f} kmol")