import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt
from tqdm import tqdm

# WARNING! This code is only to replicate the five examples in Gelmi, Jorquera.

######################################
#       1. PROBLEM PARAMETERS        #
######################################

TEST_CASE_ID = 5                                # Set to 1, 2, 3, 4 or 5.
VARIABLE_MEMORY = (TEST_CASE_ID == 4)           # True if memory integration range depends on time. For these test cases only case 4 is variable.

match TEST_CASE_ID:                             # Initial conditions.
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
    
T = 1                                           # Time horizon.

def memorylessOperator(t, y):                   # Operator to solve the ODE without memory.
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
            A = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]) 
            return A @ y + np.array([0, 0, 0, np.exp(t) - t])

def kernel(t, s):                               # Integration kernel (the integral is in the variable s).
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
        
def lowerIntegration(t):                        # Lower integration extreme in memory term.
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

def upperIntegration(t):                        # Upper integration extreme in memory term.
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

TOL = 1e-8                                              # Tolerance for error check.
NS = np.array([100]) 
# NS = np.array([50, 100, 200, 500, 1000, 2000])        # Number of mesh points.
SMOOTHING = 0.5                                         # Smoothing parameter.
ITER = 0                                                # Counter for iterations.
ERRORS = []                                             # Error array for accuracy test.
PLOT_SOLUTION = True                                    # Plot solution?
PLOT_ERRORS = False                                     # Plot error comparison?

for N in tqdm(NS):

    MESH = np.linspace(0, T, N)                         # Equispaced mesh.

    match TEST_CASE_ID:                                 # Exact solution.
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

    ######################################
    #         3. SOLVER ROUTINES         #
    ######################################

    def areSimilar(y1, y2):                             # Metric of similarity.
        return np.linalg.norm(y1 - y2) <= TOL

    def computeMemory(y):                               # Compute memory term.
        match TEST_CASE_ID:
            case 3:
                factor2 = y ** 2
            case _:
                factor2 = y
        memory = np.zeros(N)
        if VARIABLE_MEMORY:
            for i in range(1, N-1):
                lowExtreme = lowerIntegration(MESH[i])
                uppExtreme = upperIntegration(MESH[i])
                lowIdx = np.searchsorted(MESH, lowExtreme)
                uppIdx = np.searchsorted(MESH, uppExtreme)
                quadMesh = np.unique(np.concatenate((np.array([lowExtreme]),  MESH[lowIdx:uppIdx],  np.array([uppExtreme]))))
                factor1 = np.array([kernel(MESH[i], MESH[j]) for j in range(N)])
                ynew = np.interp(quadMesh, MESH, factor1 * factor2)
                memory[i] = simpson(ynew, quadMesh)
        else:
            for i in range(N):
                factor1 = np.array([kernel(MESH[i], MESH[j]) for j in range(N)])
                memory[i] = simpson(factor1 * factor2, MESH)
        match TEST_CASE_ID:
            case 5:
                def memoryOperator(t, y):
                    j = np.floor(t * N / T).astype(int)
                    if j == N:
                        j -= 1
                    return memorylessOperator(t, y) + np.array([0, 0, 0, memory[j]])
            case _:
                def memoryOperator(t, y):
                    j = np.floor(t * N / T).astype(int)
                    if j == N:
                        j -= 1
                    return memorylessOperator(t, y) + np.array([memory[j]])
        return memoryOperator

    def plotSolution(sol):                              # Routine to plot solution.
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
    
    def plotError():                                    # Routine to plot errors.
        plt.figure(figsize=(10, 4))
        plt.plot(NS, ERRORS, label='Error plot', color='blue')
        plt.plot(NS, 1 / NS, '-.', label='order 1', color='red')
        plt.plot(NS, 1 / NS ** 0.5, '-.', label='order 1/2', color='black')
        plt.title('Approximate solution vs exact solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.show()
        return

    # 3.0. Compute solution without memory
    guess = solve_ivp(memorylessOperator, [0, T], initial, method='RK45', t_eval=MESH).y[0]
    done = False

    while not done:

        # 3.1. Update counter
        ITER += 1
        # print(f'Iteration {ITER}')

        # 3.2. Compute integral term
        memoryOperator = computeMemory(guess)

        # 3.3. Compute solution with estimated memory term
        y = solve_ivp(memoryOperator, [0, T], initial, method='RK45', t_eval=MESH).y[0]

        # 3.4. Check similarity and update guess
        if areSimilar(y, guess):
            done = True
        else:
            guess = SMOOTHING * y + (1 - SMOOTHING) * guess

    ERRORS.append(np.max(np.abs(y - EXACT)))

if PLOT_SOLUTION:
    plotSolution(y)
if PLOT_ERRORS:
    plotError()