import numpy as np
from scipy.integrate import simpson, quad
import matplotlib.pyplot as plt
from integrodifferential import IntegroDifferentialProblem

class MGTGPSolver(IntegroDifferentialProblem):

    def __init__(self, initial, params):
        super().__init__(initial, **params)
        return

    def source(self, t: float) -> np.ndarray:
        """
        The function g such that y' = F(t, y) + memory(t, y) + g(t).
        """
        return np.array([0, 0, \
                     2 * (self.ProblemData.alpha + self.ProblemData.delta * self.ProblemData.mode) + 2 * self.ProblemData.beta * self.ProblemData.mode * t + self.ProblemData.gamma * self.ProblemData.mode * (t ** 2 + 0.1) - self.ProblemData.rho * self.ProblemData.mode * (0.1 * np.cos(t) + t), \
                     - 0.1 * np.sin(t) + 1 + self.ProblemData.mode * self.ProblemData.initial[-1] * np.exp(-t) + self.ProblemData.mode * (0.1 * 0.5 * (np.cos(t) + np.sin(t) - np.exp(-t)) + t - 1 + np.exp(-t)) + 2 * self.ProblemData.rho * self.ProblemData.mode + 2 * self.ProblemData.rho * self.ProblemData.gamma / self.ProblemData.beta * self.ProblemData.mode * t])

    def memoryless_operator(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Computes F(t, y), i.e. the ODE part without memory.
        """
        y2 = (- self.ProblemData.gamma * self.ProblemData.mode * y[0]
                - self.ProblemData.beta * self.ProblemData.mode * y[1]
                - (self.ProblemData.delta * self.ProblemData.mode + self.ProblemData.alpha) * y[2]
                + self.ProblemData.rho * self.ProblemData.mode * y[3])
        y3 = (- self.ProblemData.rho * self.ProblemData.mode * self.ProblemData.gamma / self.ProblemData.beta * y[1]
                - self.ProblemData.rho * self.ProblemData.mode * y[2]
                - self.kernel_integrals[0] * self.ProblemData.mode * y[3])
        return np.array([y[1], y[2], y2, y3])
    
    def kernel(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Interaction kernel to compute memory term. Expected integration with respect to s.
        """
        truncation = 35.0
        diff = t - s
        val = np.exp(- diff)
        return np.where(diff <= truncation, val, 0.0)

    def compute_memory(self, y: np.ndarray) -> np.ndarray:
        """
        Function to compute the memory term. This function directly stores values in self.memory.
        """
        factor2 = - self.ProblemData.mode * y
        for i in range(self.NumericalParameters.n_elements):
            quadMesh = self.NumericalParameters.mesh[0:i+1]
            factor1 = self.kernel(self.NumericalParameters.mesh[i], quadMesh)
            ynew = factor1 * factor2[0:i+1]
            self.memory[i] = simpson(ynew, x=quadMesh) - self.ProblemData.initial[-1] * self.ProblemData.mode * self.kernel_integrals[i] + self.kernel_integrals[0] * self.ProblemData.mode * y[i] 
        return

    def similarity_metric(self, guess: np.ndarray, solution: np.ndarray) -> float:
        """
        Function to compute the distance between old and new guesses.
        """
        return np.sum((solution[-1] - guess[-1]) ** 2)
    
    def mu(self, t: np.ndarray) -> np.ndarray:
        """
        Integral of the kernel from t to infinity. 
        """
        truncation = 35.0
        val = np.exp(-t)
        return np.where(t <= truncation, val, 0.0)
    
    def precompute_integrals(self):
        """
        Precomputes integrals of the kernel.
        """
        mesh = np.linspace(0, self.NumericalParameters.final_time, self.NumericalParameters.n_elements)
        self.kernel_integrals = np.zeros(self.NumericalParameters.n_elements)
        for i in range(self.NumericalParameters.n_elements):
            # self.kernel_integrals[i], _ = quad(lambda t: self.kernel(t, np.array([0])), mesh[i], 25)
            self.kernel_integrals[i] = np.exp(-mesh[i])
        return
    
    def exact(self, t: np.ndarray) -> np.ndarray:
        """
        Computes exact solution.
        """
        y1 = 0.1 + t ** 2
        y2 = 2 * t
        y3 = 2 * np.ones(self.NumericalParameters.n_elements)
        y4 = 0.1 * np.cos(t) + t
        return [y1, y2, y3, y4]

def main():
    # === Instantiate solver ===
    params = {'alpha': 1.0,
              'beta': 100.0,
              'gamma': 20.0,
              'delta': 1.0,
              'rho': 1.0,
              'mode': 4.0}
    initial_conditions = [0.1, 0.0, 2.0, 0.1]
    solver = MGTGPSolver(initial_conditions, params)

    # === Configure numerical parameters ===
    solver.NumericalParameters.n_elements = 1000
    solver.NumericalParameters.final_time = 50
    solver.NumericalParameters.tol = 1e-8
    solver.NumericalParameters.max_iter = 300
    solver.NumericalParameters.smoothing = 0.9
    solver.NumericalParameters.solver = "Radau"
    solver.NumericalParameters.exact = solver.exact

    # === Configure settings ===
    solver.Settings.dump_parameters = True
    solver.Settings.dump_solution = True
    solver.Settings.dump_convergence_data = True
    solver.Settings.path_prefix = '../validation/'
    solver.Settings.dump_label = 'validation'

    # === Solve ===
    solver.precompute_integrals()
    solver.solve()

    # === Plot ===
    solver.plot(save_fig=True, show_errors=True, show_exact=True, fig_size=(3.2, 3.2), names=['u', 'u_t', 'u_{tt}', '\\theta'])

if __name__ == "__main__":
    main()