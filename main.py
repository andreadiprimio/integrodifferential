import numpy as np
from scipy.integrate import simpson, quad
import matplotlib.pyplot as plt
from integrodifferential import IntegroDifferentialProblem

class Solver(IntegroDifferentialProblem):

    def __init__(self, initial, params):
        super().__init__(initial, **params)
        return

    def source(self, t: float) -> np.ndarray:
        """
        The function g such that y' = F(t, y) + memory(t, y) + g(t).
        """
        return np.array([0, 0, 0, 0])

    def memoryless_operator(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Computes F(t, y), i.e. the ODE part without memory.
        """
        return y
    
    def kernel(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Interaction kernel to compute memory term. Expected integration with respect to s.
        """
        return 0

    def similarity_metric(self, guess: np.ndarray, solution: np.ndarray) -> float:
        """
        Function to compute the distance between old and new guesses.
        """
        return np.sum((solution[-1] - guess[-1]) ** 2)
    
    def exact(self, t: np.ndarray) -> np.ndarray:
        """
        Computes exact solution.
        """
        return 
    
def main():
    # === Instantiate solver ===
    # Setup parameters that are needed. Solver methods may access them via solver.ProblemData.parameter_name.
    params = {'alpha': 1.0,
              'beta': 10.0,
              'gamma': 20.0,
              'delta': 1.0,
              'rho': 1.0,
              'mode': 1.0}
    initial_conditions = []
    solver = Solver(initial_conditions, params)

    # === Configure numerical parameters ===
    solver.NumericalParameters.n_elements = 1000
    solver.NumericalParameters.final_time = 10
    solver.NumericalParameters.tol = 1e-8
    solver.NumericalParameters.max_iter = 300
    solver.NumericalParameters.smoothing = 0.9
    solver.NumericalParameters.solver = "Radau"
    solver.NumericalParameters.exact = solver.exact

    # === Configure settings ===
    solver.Settings.dump_parameters = True
    solver.Settings.dump_solution = True
    solver.Settings.dump_convergence_data = True
    solver.Settings.path_prefix = './'
    solver.Settings.dump_label = 'test'

    # === Solve ===
    solver.solve()

    # === Plot ===
    solver.plot(save_fig=True, show_errors=True, show_exact=True)

if __name__ == "__main__":
    main()