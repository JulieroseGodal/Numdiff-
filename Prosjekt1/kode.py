import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display


from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve



def laplacian(U, h):
    ''' 
    Computes the **second-order stable Laplacian** using:
    - **Central differences for interior points** (second-order accurate).
    - **Smooth Neumann boundary conditions (`du/dn = 0`)**.
    
    Parameters:
        U: 2D NumPy array representing function values.
        h: Step size in space.

    Returns:
        lap: 2D NumPy array containing the Laplacian.
    '''
    Nx, Ny = U.shape
    lap = np.zeros_like(U)

    #Interior points: Second-order Central Differences
    lap[1:-1, 1:-1] = (
        (U[:-2, 1:-1] - 2 * U[1:-1, 1:-1] + U[2:, 1:-1]) / h**2 +  # d²U/dx²
        (U[1:-1, :-2] - 2 * U[1:-1, 1:-1] + U[1:-1, 2:]) / h**2    # d²U/dy²
    )

    #Neumann Boundary Conditions (`du/dn = 0` ensures smooth behavior)
    # Top boundary (i=0)
    lap[0, 1:-1] = (
        (U[1, 1:-1] - U[0, 1:-1]) / h**2 +  # Forward diff in y
        (U[0, 2:] - 2 * U[0, 1:-1] + U[0, :-2]) / h**2  # Central in x
    )

    # Bottom boundary (i=-1)
    lap[-1, 1:-1] = (
        (U[-2, 1:-1] - U[-1, 1:-1]) / h**2 +  # Backward diff in y
        (U[-1, 2:] - 2 * U[-1, 1:-1] + U[-1, :-2]) / h**2  # Central in x
    )

    # Left boundary (j=0)
    lap[1:-1, 0] = (
        (U[2:, 0] - 2 * U[1:-1, 0] + U[:-2, 0]) / h**2 +  # Central in y
        (U[1:-1, 1] - U[1:-1, 0]) / h**2  # Forward in x
    )

    # Right boundary (j=-1)
    lap[1:-1, -1] = (
        (U[2:, -1] - 2 * U[1:-1, -1] + U[:-2, -1]) / h**2 +  # Central in y
        (U[1:-1, -2] - U[1:-1, -1]) / h**2  # Backward in x
    )

    lap[0, 0] = lap[1, 1]
    lap[0, -1] = lap[1, -2]
    lap[-1, 0] = lap[-2, 1]
    lap[-1, -1] = lap[-2, -2]


    #Stabilization: Clip extreme values
    lap = np.clip(lap, -1e6, 1e6)  # Prevent numerical blow-up

    return lap

def S_t(S, I, beta, mu_S, h):
    return -beta * I * S + mu_S * laplacian(S, h)

def I_t(S, I, beta, mu_I, h, gamma):
    return beta * I * S - gamma * I + mu_I * laplacian(I, h)
def R_t(I, gamma): 
    return gamma * I



def population_init(Lx,Ly,h,I_A_size, I_A_pos, num_infected): 
   
    '''
    Function to initialise population
    input: 
        Lx,Ly : length of grid in x and y direction ( size of grid)
        h : steplength in  space
        pupulation_size: number of individuals in populatio
        I_A_size: size of area containing infected people (x,y)
        I_A_pos: top left corner of area of infected individuals
        perc_I, perc_S: percentage of population thats infected or susceptible respectivly
    output:
        S: initial matrix of susceptible individuals
        I: initial matrix of infected individuals
        R: initial matrix of removed individuals
    '''
    Nx, Ny = int(Lx / h), int(Ly / h)  # Grid resolution
    grid = np.array([Nx, Ny])

    # Initialize all individuals as susceptible (homogeneous)
    S = np.ones((Nx, Ny))  # Every grid cell starts as 100% susceptible
    I = np.zeros((Nx, Ny))  # No infection initially

 # Define infected area bounds
    x_start, y_start = int(I_A_pos[0] / h), int(I_A_pos[1] / h)
    x_end, y_end = x_start + int(I_A_size[0] / h), y_start + int(I_A_size[1] / h)

    #Ensure boundaries are within grid
    x_end = min(x_end, Nx)
    y_end = min(y_end, Ny)

    # Randomly distribute infected individuals within the infected area
    infected_positions = set()
    while len(infected_positions) < num_infected:
        i = np.random.randint(x_start, x_end)
        j = np.random.randint(y_start, y_end)
        if (i, j) not in infected_positions:
            infected_positions.add((i, j))
            I[i, j] = 1  #mark infected

    return S, I, grid


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def SIR_system(t, U, beta, mu_S, mu_I, gamma, h, Nx, Ny):
    total_points = Nx * Ny

    if U.ndim == 2:
        U = U[:, 0]  
    
    if U.size != 3 * total_points:
        raise ValueError(f"Incorrect U size: expected {3 * total_points}, got {U.size}")

    S = U[:total_points].reshape((Nx, Ny))
    I = U[total_points:2*total_points].reshape((Nx, Ny))
    R = U[2*total_points:].reshape((Nx, Ny))

    lap_S = laplacian(S, h)
    lap_I = laplacian(I, h)
    print("Min/Max S before diffusion:", np.min(S), np.max(S))
    print("Min/Max laplacian(S):", np.min(lap_S), np.max(lap_S))


    dSdt = -beta * I * S + mu_S * lap_S
    dIdt = beta * I * S - gamma * I + mu_I * lap_I
    dRdt = gamma * I

    return np.concatenate([dSdt.flatten(), dIdt.flatten(), dRdt.flatten()])

def solve_SIR_with_ivp(population, beta, mu_S, mu_I, gamma, h, T):
    S0, I0, grid = population
    Nx, Ny = grid
    Nt = int(T / h)
    R0 = np.zeros_like(S0)  

    U0 = np.concatenate([S0.flatten(), I0.flatten(), R0.flatten()])
    t_span = (0, T)
    t_eval = np.linspace(0, T, Nt)

    sol = solve_ivp(SIR_system, t_span, U0, t_eval=t_eval, method="RK45",
                     args=(beta, mu_S, mu_I, gamma, h, Nx, Ny))

    Nt = len(sol.t)
    
    S_sol = sol.y[:Nx * Ny, :].reshape((Nx, Ny, Nt), order="F")
    I_sol = sol.y[Nx * Ny:2 * Nx * Ny, :].reshape((Nx, Ny, Nt), order="F")
    R_sol = sol.y[2 * Nx * Ny:, :].reshape((Nx, Ny, Nt), order="F")

    return S_sol, I_sol, R_sol, Nt

def animate_SIR_plotly(S_sim, I_sim, R_sim, Nt, T, max_frames=1000):
    dt = T / Nt
    zmin, zmax = 0, 1

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Susceptible", "Infected", "Recovered"],
        horizontal_spacing=0.1
    )

    fig.add_trace(go.Heatmap(z=S_sim[:, :, 0], colorscale='Blues', showscale=True, zmin=zmin, zmax=zmax, colorbar=dict(title="Susceptible", x=0.3)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=I_sim[:, :, 0], colorscale='Magma', showscale=True, zmin=zmin, zmax=zmax, colorbar=dict(title="Infected", x=0.6)), row=1, col=2)
    fig.add_trace(go.Heatmap(z=R_sim[:, :, 0], colorscale='Greens', showscale=True, zmin=zmin, zmax=zmax, colorbar=dict(title="Recovered", x=1.0)), row=1, col=3)

    frame_step = max(1, Nt // max_frames)
    frames = []
    for frame in range(0, Nt, frame_step):
        frame_data = [
            go.Heatmap(z=S_sim[:, :, frame], colorscale='Blues', zmin=zmin, zmax=zmax, colorbar=dict(title="Susceptible", x=0.3)),
            go.Heatmap(z=I_sim[:, :, frame], colorscale='Magma', zmin=zmin, zmax=zmax, colorbar=dict(title="Infected", x=0.6)),
            go.Heatmap(z=R_sim[:, :, frame], colorscale='Greens', zmin=zmin, zmax=zmax, colorbar=dict(title="Recovered", x=1.0))
        ]
        frames.append(go.Frame(data=frame_data, name=f"t = {frame * dt:.2f} s"))

    fig.frames = frames

    fig.update_layout(
        title="SIR Simulation Over Time",
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                 "label": "Play", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                 "label": "Pause", "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "type": "buttons",
            "x": 0.1, "xanchor": "right", "y": 1.15, "yanchor": "top"
        }]
    )

    pio.show(fig)

def euler(S, I,R, beta, mu_S, mu_I, gamma, h, k):
    S_new = S + k * S_t(S, I, beta, mu_S, h)
    I_new = I + k * I_t(S, I, beta, mu_I, h, gamma)
    R_new = R + k * R_t(I, gamma)
    return S_new, I_new , R_new

def solver (population, beta, mu_S, mu_I, gamma,k, h,  T = 10 ):
    '''
    Solves spatian SIR model with fav ODE solver
        input: 
        beta, mu_S, mu_I, gamma: model parameters
        Lx,Ly: spacial domain size ( default domain is unit square)
        k: time stepsize
        h: space stepsize
        T: total simulation time

    returns: 
        S, I: final solutions in T

    
    '''
   
    Nt = int(T/k)
 
    S0, I0,grid = population
    S = np.zeros((grid[0],grid[1], Nt))
    R = np.zeros_like(S)
    I= np.zeros_like(S)
    lap_S =np.zeros_like(S)
    lap_I =  np.zeros_like(I)

    S[:,:,0] = S0
    R[:,:,0] = 0
    I[:,:,0] = I0

    for t in range(1,Nt):
        print(f"Time Step {t}/{Nt}")
        print(f"  S: min={S.min():.5f}, max={S.max():.5f}")
        print(f"  I: min={I.min():.5f}, max={I.max():.5f}")
    
        S[:,:,t] , I [:,:,t],R[:,:,t]= euler(S[:,:,t-1], I[:,:,t-1], R[:,:,t-1], beta, mu_S, mu_I, gamma, h, k)

        
    return S, I, R, Nt


def run_sim(k, h, I_A_size, I_A_pos, num_infected, beta, mu_S, mu_I, gamma, Lx=1, Ly=1, T=100):
    ''' 
    Function to run simulation with given parameters and display animation inline.
    
    h : space step size.
    k : time step size.
    population_size : number of individuals in the population.
    I_A_size : tuple (x, y) size of area containing infected people.
    I_A_pos : tuple (x, y) top left corner of the infected individuals area.
    perc_I, perc_S : percentage of population that's infected or susceptible, respectively.
    beta, mu_S, mu_I, gamma : model parameters.
    Lx, Ly : spatial domain size (default is unit square).
    T : total simulation time.
    '''
    # Initialize the population on a spatial grid
    population = population_init(Lx,Ly,h,I_A_size, I_A_pos, num_infected)
    
    # Run the solver (which should return 3D arrays for S, I, and R and the total number of time steps)
    
    # Run the solver (which should return 3D arrays for S, I, and R and the total number of time steps)
    S_sim, I_sim, R_sim, Nt= solve_SIR_with_ivp(population, beta, mu_S, mu_I, gamma, h, T)#solver(population, beta, mu_S, mu_I, gamma, k, h, T)
    
    # Create the animation object
    # animate_SIR(S_sim, I_sim, R_sim, Nt, Lx, Ly, T)
    print("ANIMATE")
    animate_SIR_plotly(S_sim, I_sim, R_sim,Nt, T)
    
    # Return the animation as an HTML object to display inline in Jupyter Notebook
    #return 
# from IPython.display import display



run_sim(
    k=0.01**2/10, 
    h=0.08,

    I_A_size=(1, 1),
    I_A_pos=(0.45, 0.45),
    num_infected=1,
    beta=1,
    mu_S = 0.001,
    mu_I=0.01,
    gamma=0.5,
    Lx=5,
    Ly=5,
    T=40)
# display(ani_html)

print(3)