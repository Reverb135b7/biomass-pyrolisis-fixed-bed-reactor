"""
Streamlit Dashboard for Biomass Pyrolysis Fixed-Bed Reactor Simulation
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))
from pyrolysis_model import run_simulation, extract_results, default_params, calculate_kinetics

st.set_page_config(
    page_title="Biomass Pyrolysis Reactor",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Biomass Pyrolysis Fixed-Bed Reactor Simulator")
st.markdown("### Dynamic Multiphase Modeling of Primary & Secondary Pyrolysis Reactions")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Simulation Parameters")

# Spatial discretization
N = st.sidebar.slider("Spatial grid points (N)", 10, 200, 50, 10)
L = st.sidebar.slider("Reactor length [m]", 0.5, 2.0, 1.0, 0.1)

# Kinetic parameters section
st.sidebar.subheader("Heating Rate Control")
heating_rate = st.sidebar.slider("Effective heating rate factor", 0.5, 3.0, 1.0, 0.1)

# Gas flow parameters
st.sidebar.subheader("Gas Flow Parameters")
u_g = st.sidebar.slider("Superficial gas velocity u_g [m/s]", 0.01, 1.0, 0.1, 0.01)
D_ax = st.sidebar.slider("Axial dispersion D_ax [m²/s]", 0.001, 0.5, 0.01, 0.001)
d_p = st.sidebar.slider("Particle diameter d_p [mm]", 0.5, 5.0, 2.0, 0.1) / 1000  # convert to m
phi = st.sidebar.slider("Sphericity φ", 0.6, 1.0, 0.85, 0.01)
mu = st.sidebar.slider("Gas viscosity μ [Pa·s]", 1e-5, 1e-3, 2.5e-5, 1e-6)

# Temperature parameters
st.sidebar.subheader("Temperature Profile")
T_initial = st.sidebar.slider("Initial temperature T_0 [K]", 300, 600, 300)
T_heat = st.sidebar.slider("Maximum reactor temperature [K]", 500, 800, 700)

# Time parameters
t_max = st.sidebar.slider("Simulation time [s]", 60, 1800, 600, 60)

# ============================================================
# BUILD PARAMETERS DICT
# ============================================================

params = default_params()
params['N'] = N
params['L'] = L
params['dz'] = L / (N - 1)
params['u_g'] = u_g
params['D_ax'] = D_ax
params['d_p'] = d_p
params['phi'] = phi
params['mu'] = mu
params['T_s0'] = T_initial
params['T_g0'] = T_initial
params['t_max'] = t_max

# Adjust effective conductivity based on heating rate
params['k_eff'] = 0.5 * heating_rate
params['k_g_eff'] = 0.03 * heating_rate
params['h_v_a_v'] = 50.0 * heating_rate

# Set initial biomass density based on temperature (higher initial temp → faster devolatilization)
params['rho_s0'] = 500.0 * (1 - (T_initial - 300) / 400 * 0.5)  # decrease with higher initial temp
params['rho_s0'] = max(params['rho_s0'], 100.0)  # minimum

# ============================================================
# RUN SIMULATION
# ============================================================

run_button = st.sidebar.button("▶️ Run Simulation", type="primary")

if run_button:
    with st.spinner("Solving multiphase pyrolysis model..."):
        try:
            sol, final_params = run_simulation(params)
            results = extract_results(sol, final_params)
            
            st.success("✅ Simulation completed successfully!")
            
            # ============================================================
            # DISPLAY RESULTS
            # ============================================================
            
            tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Temperature Profiles", "📊 Concentration Profiles", "📈 Yields Over Time", "🔍 Parameter Effects"])
            
            with tab1:
                st.subheader("Axial Temperature Profiles")
                t = results['time']
                z = results['z']
                N = results['N']
                
                # Find indices for display
                idx_start = 0
                idx_mid = len(t) // 2
                idx_end = len(t) - 1
                
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                
                # Solid temperature
                ax1.plot(z, results['T_s'][idx_start, :], 'b-', label=f'Solid t={t[idx_start]:.1f}s', alpha=0.7)
                ax1.plot(z, results['T_s'][idx_mid, :], 'b-', label=f'Solid t={t[idx_mid]:.1f}s', alpha=0.7)
                ax1.plot(z, results['T_s'][idx_end, :], 'b-', label=f'Solid t={t[idx_end]:.1f}s', linewidth=2)
                
                # Gas temperature
                ax1.plot(z, results['T_g'][idx_start, :], 'r--', label=f'Gas t={t[idx_start]:.1f}s', alpha=0.7)
                ax1.plot(z, results['T_g'][idx_mid, :], 'r--', label=f'Gas t={t[idx_mid]:.1f}s', alpha=0.7)
                ax1.plot(z, results['T_g'][idx_end, :], 'r--', label=f'Gas t={t[idx_end]:.1f}s', linewidth=2)
                
                ax1.set_xlabel('Reactor length z [m]')
                ax1.set_ylabel('Temperature [K]')
                ax1.set_title('Solid & Gas Temperature Profiles at Different Times')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(280, 850)
                
                st.pyplot(fig1)
                
            with tab2:
                st.subheader("Species Concentration Profiles")
                fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Tar concentration
                idx_mid = len(t) // 2
                idx_end = len(t) - 1
                
                ax2a.plot(z, results['C_tar'][idx_mid, :], 'm-', label='Tar', linewidth=2)
                ax2a.plot(z, results['C_tar'][idx_end, :], 'm--', label='Tar (final)', linewidth=2)
                ax2a.set_xlabel('Reactor length z [m]')
                ax2a.set_ylabel('Tar concentration [kmol/m³]')
                ax2a.set_title('Tar Concentration Profile')
                ax2a.legend()
                ax2a.grid(True, alpha=0.3)
                ax2a.set_ylim(bottom=0)
                
                # Gas species (simplified: total gases)
                ax2b.plot(z, results['C_gases'][idx_end, :], 'c-', label='Gases', linewidth=2)
                ax2b.set_xlabel('Reactor length z [m]')
                ax2b.set_ylabel('Gas concentration [kmol/m³]')
                ax2b.set_title('Gas Species Concentration')
                ax2b.legend()
                ax2b.grid(True, alpha=0.3)
                ax2b.set_ylim(bottom=0)
                
                st.pyplot(fig2)
                
            with tab3:
                st.subheader("Integrated Yields Over Time")
                fig3, ax4 = plt.subplots(figsize=(10, 5))
                
                t_min = results['time'] / 60  # convert to minutes
                ax4.plot(t_min, results['tar_yield'], 'mo-', label='Tar yield', linewidth=2, markersize=4)
                ax4.plot(t_min, results['gases_yield'], 'co-', label='Gases yield', linewidth=2, markersize=4)
                
                ax4.set_xlabel('Time [minutes]')
                ax4.set_ylabel('Yield [kmol]')
                ax4.set_title('Integrated Yields vs. Time')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(bottom=0)
                ax4.annotate(f'Final tar: {results["tar_yield"][-1]:.3f} kmol', 
                           xy=(t_min[-1], results['tar_yield'][-1]),
                           xytext=(0.6, 0.9), textcoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                st.pyplot(fig3)
                
            with tab4:
                st.subheader("Key Simulation Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Reactor Configuration**")
                    st.write(f"• Length: {final_params['L']} m")
                    st.write(f"• Grid points: {final_params['N']}")
                    st.write(f"• Particle diameter: {final_params['d_p']*1000:.1f} mm")
                    st.write(f"• Voidage: {final_params['epsilon']}")
                    st.write(f"• Gas velocity: {final_params['u_g']} m/s")
                    
                with col2:
                    st.markdown("**Kinetic & Thermal**")
                    st.write(f"• Heating factor: {heating_rate}")
                    st.write(f"• Max temp: {T_heat} K")
                    st.write(f"• Initial temp: {T_initial} K")
                    st.write(f"• Dispersion: {D_ax:.4f} m²/s")
                    st.write(f"• Viscosity: {mu:.2e} Pa·s")
                
                # Summary metrics
                st.markdown("**Summary Results**")
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("Final Tar Yield", f"{results['tar_yield'][-1]:.4f} kmol")
                with col4:
                    st.metric("Final Gases Yield", f"{results['gases_yield'][-1]:.4f} kmol")
                with col5:
                    temp_rise = np.max(results['T_s'][-1, :]) - T_initial
                    st.metric("Max Temp Rise", f"{temp_rise:.1f} K")
                    
        except Exception as e:
            st.error(f"❌ Simulation error: {str(e)}")
            st.info("Try adjusting parameters (reduce grid points, change temperature range, etc.)")
            import traceback
            st.code(traceback.format_exc())
else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start the analysis.")
    st.markdown("""
    **Project 5: Dynamic Multiphase Modeling of Biomass Pyrolysis & Secondary Tar Cracking**
    
    This simulator implements the Method of Lines (MOL) approach using `scipy.integrate.solve_ivp`
    with Radau solver for stiff ODE systems. The model tracks:
    
    - **Solid phase**: Multi-reaction devolatilization (hemicellulose, cellulose, lignin)
    - **Gas phase**: Advection, dispersion, and secondary tar cracking
    - **Interphase heat transfer**: Convective coupling between solid and gas
    - **Fluid dynamics**: Ergun equation for pressure drop
    """)