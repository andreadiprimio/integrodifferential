import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from tqdm.rich import tqdm
from numba import njit

# WARNING! This code is adapted to replicate the five examples in Gelmi, Jorquera.

######################################
#       1. PROBLEM PARAMETERS        #
######################################

# Following the order in Gelmi, Jorquera, set to 1, 2, 3, 4 or 5.
TEST_CASE_ID = 1

# Set to True if memory integration range depends on time. For these test cases only case 4 is variable.
VARIABLE_MEMORY = (TEST_CASE_ID == 4)

# Initial conditions based on the test case ID.
match TEST_CASE_ID:
    case 1:
        initial = np.array([0])
    case 2:
        initial = np.array([1])
    case 3:
        initial = np.array([1])
    case 4:
        initial = np.array([1])
    case 5:
        initial = np.array([1, 1, 1, 1])
    case _:
        raise ValueError

# Time horizon.
T = 1  

# Operator to solve the ODE without memory.
@njit
def memorylessOperator(t, y):
    match TEST_CASE_ID:
        case 1:
            return y - 0.5 * t + 1 / (1 + t) - np.log(1 + t)
        case 2:
            return y - np.cos(2 * np.pi * t) - 2 * np.pi * np.sin(2 * np.pi * t) - 0.5 * np.sin(4 * np.pi * t)
        case 3:
            return 1 - 29 / 60 * t
        case 4:
            return t * (1 + np.sqrt(t)) * np.exp(-np.sqrt(t)) - (t ** 2 + t + 1) * np.exp(-t)
        case 5:
            return np.array([y[1], y[2], y[3], np.exp(t) - t])

# Integration kernel (the integral is in the variable s).
@njit
def kernel(t, s):
    match TEST_CASE_ID:
        case 1:
            return 1 / (np.log(2) ** 2) * t / (1 + s)
        case 2:
            return np.sin(4 * np.pi * t + 2 * np.pi * s)
        case 3:
            return t * s
        case 4:
            return t * s
        case 5:
            return t * s

# Lower integration extreme in memory term.
def lowerIntegration(t):
    match TEST_CASE_ID:
        case 1:
            return 0
        case 2:
            return 0
        case 3:
            return 0
        case 4:
            return t
        case 5:
            return 0

# Upper integration extreme in memory term.
def upperIntegration(t):
    match TEST_CASE_ID:
        case 1:
            return 1
        case 2:
            return 1
        case 3:
            return 1
        case 4:
            return np.sqrt(t)
        case 5:
            return 1

######################################
#      2. NUMERICAL PARAMETERS       #
######################################

# Tolerance for error check.
TOL = 1e-8

# Number of mesh points, for accuracy testing multiple values are needed.
# NS = np.array([500])
NS = np.array([50, 100, 200, 500, 1000, 2000, 5000])

# Smoothing parameter.
SMOOTHING = 0.5

# Maximum number of iterations.
MAX_ITER = 200

# Counter for iterations.
ITER = 0

# Error array for accuracy test.
ERRORS = []

# Plotting flags.
PLOT_SOLUTION = True  
PLOT_ERRORS = True 
PLOT_CONVERGENCE = False

# Plotting routines.
def plotSolution(sol):
        plt.figure(figsize=(10, 4))
        plt.plot(MESH, sol, label='Solution plot', color='blue')
        plt.plot(MESH, EXACT, '-.', label='Exact solution', color='red')
        plt.title('Approximate solution vs exact solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.show()
        return

def plotError():
    plt.figure(figsize=(10, 4))
    plt.plot(NS, ERRORS, label='Error plot', color='blue')
    plt.plot(NS, 1 / NS, '-.', label='order 1', color='red')
    plt.plot(NS, 1 / NS ** 0.5, '-.', label='order 1/2', color='black')
    for i in range(1, len(ERRORS)):
        accuracyOrder = (np.log(ERRORS[i]) - np.log(ERRORS[i-1])) / (np.log(1 / NS[i]) - np.log(1 / NS[i-1]))
        plt.text((NS[i] + NS[i-1]) / 2, ERRORS[i] * 1.7, f"{accuracyOrder:.2f}", fontsize=8, ha='center') # TO DO: maybe find a better way to display the order of accuracy
    plt.title('Error vs number of mesh points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xscale('log')
    plt.yscale('log')
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

for N in tqdm(NS):

    # Equispaced mesh.
    MESH = np.linspace(0, T, N)

    # Exact solution based on the test case ID.
    match TEST_CASE_ID:
        case 1:
            EXACT = np.log(1 + MESH)
        case 2:
            EXACT = np.cos(2 * np.pi * MESH)
        case 3:
            EXACT = 1 + MESH + MESH ** 2
        case 4:
            EXACT = np.exp(-MESH)
        case 5:
            EXACT = np.exp(MESH)

    # Compute memory term.
    def computeMemory(y):
        if TEST_CASE_ID == 3:
            factor2 = y ** 2
        else:
            factor2 = y

        memory = np.zeros(N)

        # Vectorized computation for non-variable memory cases
        if not VARIABLE_MEMORY:
            # Pre-compute all kernel values at once
            t_grid, s_grid = np.meshgrid(MESH, MESH, indexing='ij')
            kernel_matrix = kernel(t_grid, s_grid)
            integrand = kernel_matrix * factor2[np.newaxis, :]

            # Use vectorized simpson integration
            for i in range(N):
                memory[i] = simpson(integrand[i], x=MESH)
        else:
            # Keep original approach for variable memory
            for i in range(N):
                lowExtreme = lowerIntegration(MESH[i])
                uppExtreme = upperIntegration(MESH[i])
                assert lowExtreme <= uppExtreme, "Lower extreme must be less than or equal to upper extreme."
                if lowExtreme != uppExtreme:
                    lowIdx = np.searchsorted(MESH, lowExtreme)
                    uppIdx = np.searchsorted(MESH, uppExtreme)
                    quadMesh = np.unique(
                        np.concatenate((np.array([lowExtreme]), MESH[lowIdx:uppIdx], np.array([uppExtreme]))))
                    factor1 = np.array([kernel(MESH[i], MESH[j]) for j in range(N)])
                    ynew = np.interp(quadMesh, MESH, factor1 * factor2)
                    memory[i] = simpson(ynew, x=quadMesh)

        # Pre-allocate memory lookup for faster access
        memory_interp = lambda t_val: memory[min(int(np.floor(np.clip(t_val, 0, T) * (N - 1) / T)), N - 1)]

        if TEST_CASE_ID == 5:
            def memoryOperator(t, y):
                mem_val = memory_interp(t)
                return memorylessOperator(t, y) + np.array([0, 0, 0, mem_val])
        else:
            def memoryOperator(t, y):
                mem_val = memory_interp(t)
                return memorylessOperator(t, y) + np.array([mem_val])

        return memoryOperator

    # Reset iteration counter for each N
    ITER = 0

    # 3.0. Compute solution without memory (use same solver settings as main loop)
    guess = solve_ivp(memorylessOperator,
                               [0, T],
                               initial,
                               method='LSODA',
                               t_eval=MESH,
                               rtol=1e-8,
                               atol=1e-10).y[0]
    
    done = False
    iterationErrors = []  # Track convergence

    while not done and ITER < MAX_ITER:

        # 3.1. Update counter
        ITER += 1

        # 3.2. Compute integral term
        memoryOperator = computeMemory(guess)

        # 3.3. Compute solution with estimated memory term
        solution = solve_ivp(memoryOperator,
                             [0, T],
                             initial,
                             method='LSODA',
                             t_eval=MESH,
                             rtol=1e-8,
                             atol=1e-10).y[0]

        # 3.4. Check similarity and update guess
        currentError = np.sum((solution - guess) ** 2)
        iterationErrors.append(currentError)

        if currentError <= TOL:
            done = True
            print(f"[N = {N}] Converged after {ITER} iterations with error {currentError:.2e}")
        else:
            guess = SMOOTHING * solution + (1 - SMOOTHING) * guess

    # Check if maximum iterations reached without convergence
    if ITER >= MAX_ITER:
        print(f"Warning: Maximum iterations ({MAX_ITER}) reached without convergence for N = {N}")
        print(f"Final error: {iterationErrors[-1]:.2e}")

    if PLOT_CONVERGENCE and len(iterationErrors) > 1:
        plotConvergence(iterationErrors)

    # Max absolute difference for final error
    ERRORS.append(np.max(np.abs(solution - EXACT)))

if PLOT_SOLUTION:
    plotSolution(solution)
if PLOT_ERRORS:
    plotError()