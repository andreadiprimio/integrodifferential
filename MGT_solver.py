import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from tqdm.rich import tqdm
from numba import jit, njit

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = 3.5, 3.5
plt.rcParams["savefig.bbox"] = 'tight'
plt.rcParams["savefig.dpi"] = 'figure'
plt.rcParams["savefig.format"] = 'pdf'

######################################
#       1. PROBLEM PARAMETERS        #
######################################

# Problem parameters.
alpha = 1.0
beta = 100.0
gamma = 20.0
delta = 1.0
rho = 1.0

# Initial conditions.
initial = [0.1, 0.1, 0.1, 0.1]

# Energy modes.
modes = [2] # np.arange(1, 5, 1)

# Time horizon.
T = 50

# Truncation parameter.
TRUNCATION = 40.0

# Operator to solve the ODE without memory.
# Careful: JIT compilation with Numba causes issues.
def memorylessOperator(t, y):
    val2 = - gamma * mode * y[0] - beta * mode * y[1] - (delta * mode + alpha) * y[2] + rho * mode * y[3]
    val3 = - rho * mode * gamma / beta * y[1] - rho * mode * y[2] - mode * y[3]
    return np.array([y[1], y[2], val2, val3]) + source(t)

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

# Source term at RHS. Should return a vector of size L, with L being the sum of the orders of the equations.
def source(t):
     return np.array([0, 0, \
                     2 * (alpha + delta * mode) + 2 * beta * mode * t + (gamma) * mode * (t ** 2 + 0.1) - rho * mode * (0.1 * np.cos(t) + t), \
                     # 2 * t + mode * initial[-1] * np.exp(-t) + mode * ((t ** 2 - 2 * t + 2.1) - 2.1 * np.exp(-t)) + 2 * rho * mode + 2 * rho * gamma / beta * mode * t
                     - 0.1 * np.sin(t) + 1 + mode * initial[-1] * np.exp(-t) + mode * (0.1 * 0.5 * (np.cos(t) + np.sin(t) - np.exp(-t)) + t - 1 + np.exp(-t)) + 2 * rho * mode + 2 * rho * gamma / beta * mode * t])

# Compute energy relative to mode, assuming theta(s) = theta(0) for all s < 0.
def computeEnergy(vectorSolution, mode):
    u = vectorSolution[0]
    v = vectorSolution[1]
    w = vectorSolution[2]
    theta = vectorSolution[3]
    energy = (1 + mode) * u ** 2 + (1 + mode) * v ** 2 + w ** 2 + theta ** 2
    # energy = 0.5 * (beta * (1 + mode) * (v + gamma / beta * u) ** 2 + (w + gamma / beta * v) ** 2 + gamma * alpha / beta ** 2 * (beta - gamma / alpha) * v ** 2 + delta * gamma / beta * (1 + mode) * v ** 2 + theta ** 2)
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
    # energy += 0.5 * eta
    return energy

######################################
#      2. NUMERICAL PARAMETERS       #
######################################

# Test mode? Set True only if the exact solution is known.
TEST_MODE = True

# Dump plots? Set True to save results to file.
DUMP_PLOTS = True

# Number of mesh points.
N = 500

# Tolerance for error check.
TOL = 1e-8

# Smoothing parameter.
SMOOTHING = 0.9

# Maximum number of iterations.
MAX_ITER = 200

# Counter for iterations.
ITER = 0

# Energy array for each mode.
ENERGIES = []

# Plotting flags.
PLOT_SOLUTION = True  
PLOT_CONVERGENCE = False
PLOT_ENERGY = False

# Plotting routines.
def plotEnergy():
    for i in range(len(ENERGIES)):
        plt.semilogy(MESH, ENERGIES[i], 'b-o', label='Energy', markersize=2)
        plt.title(f'Energy over time (Mode {modes[i]})')
        plt.xlabel('t')
        plt.ylabel('Energy (log scale)')
        plt.grid(True)
        plt.legend()
        if DUMP_PLOTS:
            plt.savefig(f'energy_mode_{modes[i]}.pdf')
            plt.close()
        else:
            plt.show()
    return

def plotSolution(vectorSolution):
    plt.plot(MESH, vectorSolution[0], label='Solution plot ($u$)', color='blue')
    if TEST_MODE:
        plt.plot(MESH, EXACT_U, '-.', label='Exact solution', color='red')
        plt.title('Approximate solution vs exact solution ($u$)')
    else:
        plt.title('Approximate solution ($u$)')
    plt.xlabel('$t$')
    plt.ylabel('$u$')
    plt.grid(True)
    plt.legend()
    if DUMP_PLOTS:
        plt.savefig(f'solution_u.pdf')
        plt.close()
    else:
        plt.show()

    if TEST_MODE:
        plt.semilogy(MESH, np.abs(vectorSolution[0] - EXACT_U), label='Error ($u$)', color='blue')
        plt.title('Error over time ($u$)')
        plt.xlabel('$t$')
        plt.ylabel('$|u_{exact} - u|$')
        plt.grid(True)
        plt.legend()
        if DUMP_PLOTS:
            plt.savefig(f'error_u.pdf')
            plt.close()
        else:
            plt.show()

    plt.plot(MESH, vectorSolution[3], label='Solution plot ($\\theta$)', color='blue')
    if TEST_MODE:
        plt.plot(MESH, EXACT_THETA, '-.', label='Exact solution', color='red')
        plt.title('Approximate solution vs exact solution ($\\theta$)')
    else:
        plt.title('Approximate solution ($\\theta$)')
    plt.xlabel('$t$')
    plt.ylabel('$\\theta$')
    plt.grid(True)
    plt.legend()
    if DUMP_PLOTS:
        plt.savefig(f'solution_theta.pdf')
        plt.close()
    else:
        plt.show()

    if TEST_MODE:
        plt.semilogy(MESH, np.abs(vectorSolution[3] - EXACT_THETA), label='Error ($\\theta$)', color='blue')
        plt.title('Error over time ($\\theta$)')
        plt.xlabel('$t$')
        plt.ylabel('$|\\theta_{exact} - \\theta|$')
        plt.grid(True)
        plt.legend()
        if DUMP_PLOTS:
            plt.savefig(f'error_theta.pdf')
            plt.close()
        else:
            plt.show()
    return

def plotConvergence(iterationErrors):
    plt.semilogy(iterationErrors, 'b-o', label='Iteration error', markersize=4)
    plt.title('Convergence: Error vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Error (log scale)')
    plt.grid(True)  
    plt.legend()
    if DUMP_PLOTS:
        plt.savefig(f'convergence_{mode}.pdf')
        plt.close()
    else:
        plt.show()
    return

def dumpParams():
    np.savetxt('params.txt', np.array([N, T, TOL, alpha, beta, gamma, delta, rho, mode]), header='N, T, TOL, alpha, beta, gamma, delta, rho, lambda')
    return

######################################
#         3. SOLVER ROUTINES         #
######################################

for mode in tqdm(modes):

    stability = beta - gamma / (alpha + delta * mode)
    print(f'[mode = {mode}] Stability parameter: {stability}')

    # Equispaced mesh.
    MESH = np.linspace(0, T, N)

    # Exact solutions. Only required if TEST_MODE = True.
    if TEST_MODE:
        EXACT_U = 0.1 + MESH ** 2
        EXACT_THETA = 0.1 * np.cos(MESH) + MESH

    def computeMemory(y):

        memory = np.zeros(N)
        factor2 = - mode * y

        for i in range(0, N):
            quadMesh = MESH[0:i+1]
            factor1 = np.array([kernel(MESH[i], point) for point in quadMesh])
            ynew = factor1 * factor2[0:i+1]
            memory[i] = simpson(ynew, x=quadMesh) - initial[-1] * mode * np.exp(- MESH[i]) + mode * y[i] 

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
                        method='Radau',
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
                                    method='Radau',
                                    t_eval=MESH,
                                    rtol=1e-8,
                                    atol=1e-10).y
        solution = vectorSolution[3]

        # 3.4. Check similarity and update guess
        currentError = np.sum((solution - guess) ** 2)
        print(f'[mode = {mode}] currentError = {currentError}') # temporary
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
    plotEnergy()

if PLOT_SOLUTION:
    plotSolution(vectorSolution)

if DUMP_PLOTS:
    dumpParams()