import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from tqdm.rich import tqdm
from numba import jit, njit

######################################
#       1. PROBLEM PARAMETERS        #
######################################

# Problem parameters.
alpha = 1.0
beta = 1.0
gamma = 2.0
delta = 1.0
rho = 1.0

# Initial conditions.
initial = [0.1, 0.1, 0.1, 0.1]

# Energy mode.
modes = np.arange(1, 6)

# Time horizon.
T = 1000

# Truncation parameter.
TRUNCATION = 20.0

# Operator to solve the ODE without memory.
# @njit # Careful: JIT compilation with Numba causes issues with array operations here
def memorylessOperator(t, y):
    val2 = - gamma * mode * y[0] - beta * mode * y[1] - (delta * mode + alpha) * y[2] + rho * mode * y[3]
    val3 = - rho * mode * gamma / beta * y[1] - rho * mode * y[2] - mode * y[3]    
    return np.array([y[1], y[2], val2, val3])

# Integration kernel (the integral is in the variable s). This is the kernel g.
@njit
def kernel(t, s):
    diff = t - s
    val = np.exp(- diff)
    return np.where(np.abs(diff) <= TRUNCATION, val, 0.0)

# Auxiliary kernel: the kernel g is obtaining as the integral of mu from t to infty.
@njit
def mu(t):
    val = np.exp(- t)
    return np.where(np.abs(t) <= TRUNCATION, val, np.zeros(np.size(t)))

# Compute energy relative to mode, assuming theta(s) = theta(0) for all s < 0.
def computeEnergy(vectorSolution, mode):
    u = vectorSolution[0]
    v = vectorSolution[1]
    w = vectorSolution[2]
    theta = vectorSolution[3]
    energy = (1 + mode) * u ** 2 + (1 + mode) * v ** 2 + w ** 2 + theta ** 2
    eta = np.zeros(N) # weighted norm
    kernelVals = mu(MESH) # pre-compute all values
    for i in range(N): # fix t
        quadMesh1 = MESH[0:i+1] # range [0, t]
        quadMesh2 = MESH[i:N] # range [t, T] (ideally infty, should be up to TRUNCATION, nothing changes if TRUNCATION < T)
        integrand1 = np.zeros(i+1) # maybe better names?
        integrand2 = np.zeros(N-i)
        for j in range(i+1):
            integrand1[j] = (1 + mode) * (simpson(theta[i-j:i+1], x=MESH[i-j:i+1]) ** 2) * kernelVals[j]
        thetaIntegral = simpson(theta[0:i+1], x=quadMesh1) # integral of theta from 0 to t
        for j in range(i, N):
            integrand2[j-i] = (1 + mode) * (thetaIntegral + initial[-1] * (quadMesh2[j-i] - MESH[i])) ** 2 * kernelVals[j]
        eta[i] = simpson(integrand1, x=quadMesh1) + simpson(integrand2, x=quadMesh2)
    energy += eta
    return energy

######################################
#      2. NUMERICAL PARAMETERS       #
######################################

# Tolerance for error check.
TOL = 1e-8

# Number of mesh points
N = 1000

# Smoothing parameter.
SMOOTHING = 0.9

# Maximum number of iterations.
MAX_ITER = 200

# Counter for iterations.
ITER = 0

# Energy array for each mode.
ENERGIES = []

# Plotting flags.
PLOT_SOLUTION = False  
PLOT_CONVERGENCE = False
PLOT_ENERGY = True

# Plotting routines.
def plotEnergy():
    for i in range(len(ENERGIES)):
        plt.figure(figsize=(10, 4))
        plt.semilogy(MESH, ENERGIES[i], 'b-o', label='Energy', markersize=2)
        plt.title(f'Energy over time (Mode {modes[i]})')
        plt.xlabel('Time')
        plt.ylabel('Energy (log scale)')
        plt.grid(True)
        plt.legend()
        plt.show()
    return

def plotConvergence(iterationErrors):
    plt.figure(figsize=(10, 4))
    plt.semilogy(iterationErrors, 'b-o', label='Iteration error', markersize=4)
    plt.title('Convergence: Error vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.grid(True)  
    plt.legend()
    plt.show()
    return

######################################
#         3. SOLVER ROUTINES         #
######################################

for mode in tqdm(modes):

    stability = beta - gamma / (alpha + delta * mode)
    print(f'[mode = {mode}] Stability parameter: {stability}')

    # Equispaced mesh.
    MESH = np.linspace(0, T, N)

    def computeMemory(y):

        memory = np.zeros(N)
        factor2 = - mode * y

        for i in range(0, N):
            quadMesh = MESH[0:i+1]
            factor1 = np.array([kernel(MESH[i], point) for point in quadMesh])
            ynew = factor1 * factor2[0:i+1]
            memory[i] = simpson(ynew, x=quadMesh) + mode * y[i] - initial[-1] * mode * np.exp(-MESH[i])

        memory_interp = lambda t_val: memory[min(int(np.floor(np.clip(t_val, 0, T) * (N - 1) / T)), N - 1)]
        # memory_interp = lambda t_val: np.interp(t_val, MESH, memory)

        def memoryOperator(t, y):
            mem_val = memory_interp(t)
            return memorylessOperator(t, y) + np.array([0, 0, 0, mem_val])
        
        return memoryOperator

    # Reset iteration counter for each mode
    ITER = 0

    # 3.0. Compute solution without memory (use same solver settings as main loop)
    guess = solve_ivp(memorylessOperator,
                        [0, T],
                        initial,
                        method='LSODA',
                        t_eval=MESH,
                        rtol=1e-8,
                        atol=1e-10).y[3]
    
    done = False
    iterationErrors = []  # Track convergence

    while not done and ITER < MAX_ITER:

        # 3.1. Update counter
        ITER += 1

        # 3.2. Compute integral term
        memoryOperator = computeMemory(guess)

        # 3.3. Compute solution with estimated memory term
        vectorSolution = solve_ivp(memoryOperator,
                                    [0, T],
                                    initial,
                                    method='LSODA',
                                    t_eval=MESH,
                                    rtol=1e-8,
                                    atol=1e-10).y
        solution = vectorSolution[3]

        # 3.4. Check similarity and update guess
        currentError = np.sum((solution - guess) ** 2)
        iterationErrors.append(currentError)

        if currentError <= TOL:
            done = True
            print(f"[mode = {mode}] Converged after {ITER} iterations with error {currentError:.2e}")
        else:
            guess = SMOOTHING * solution + (1 - SMOOTHING) * guess

        # Check if maximum iterations reached without convergence
        if ITER >= MAX_ITER:
            print(f"Warning: Maximum iterations ({MAX_ITER}) reached without convergence for mode = {mode}")
            print(f"Final error: {iterationErrors[-1]:.2e}")

    if PLOT_CONVERGENCE and len(iterationErrors) > 1:
        plotConvergence(iterationErrors)

    energy = computeEnergy(vectorSolution, mode)
    ENERGIES.append(energy)

if PLOT_ENERGY:
    # modes = [1, 2] # select modes to plot
    plotEnergy()