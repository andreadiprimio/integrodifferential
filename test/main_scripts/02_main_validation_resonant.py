import numpy as np
from scipy.integrate import simpson, quad
import matplotlib.pyplot as plt
from tqdm.rich import tqdm
from integrodifferential import IntegroDifferentialProblem

class MGTGPSolver(IntegroDifferentialProblem):

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
        diff = (t - s)
        val = 2 ** (-np.ceil(diff)) * (np.ceil(diff) - diff + 1)
        return val

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
        The function such that kernel(t) = integral(mu, t, infinity).
        """
        return 2 ** (-np.ceil(t))
    
    def compute_energy(self) -> np.ndarray:
        """
        Computes energy relative to mode, assuming theta(s) = theta(0) for all s < 0.
        """
        u = self.Output.solution[0]
        v = self.Output.solution[1]
        w = self.Output.solution[2]
        theta = self.Output.solution[3]
        energy = (1 + self.ProblemData.mode) * u ** 2 + (1 + self.ProblemData.mode) * v ** 2 + w ** 2 + theta ** 2
        energy_correct = 0.5 * (self.ProblemData.beta * (1 + self.ProblemData.mode) * (v + self.ProblemData.gamma / self.ProblemData.beta * u) ** 2 + (w + self.ProblemData.gamma / self.ProblemData.beta * v) ** 2 + self.ProblemData.gamma * self.ProblemData.alpha / self.ProblemData.beta ** 2 * (self.ProblemData.beta - self.ProblemData.gamma / self.ProblemData.alpha) * v ** 2 + self.ProblemData.delta * self.ProblemData.gamma / self.ProblemData.beta * (1 + self.ProblemData.mode) * v ** 2 + theta ** 2)
        eta = np.zeros(self.NumericalParameters.n_elements) # weighted norm
        mu_vals = self.mu(self.NumericalParameters.mesh) # pre-compute all values
        for i in range(self.NumericalParameters.n_elements): # fix t
            quad_mesh_1 = self.NumericalParameters.mesh[0:i+1] # range [0, t]
            quad_mesh_2 = self.NumericalParameters.mesh[i:self.NumericalParameters.n_elements] # range [t, T] (ideally infty, should be up to TRUNCATION, nothing changes if TRUNCATION < T)
            integrand_1 = np.zeros(i+1) # maybe better names?
            integrand_2 = np.zeros(self.NumericalParameters.n_elements-i)
            for j in range(i+1):
                integrand_1[j] = (1 + self.ProblemData.mode) * (simpson(theta[i-j:i+1], x=self.NumericalParameters.mesh[i-j:i+1]) ** 2) * mu_vals[j]
            thetaIntegral = simpson(theta[0:i+1], x=quad_mesh_1) # integral of theta from 0 to t
            for j in range(i, self.NumericalParameters.n_elements):
                integrand_2[j-i] = (1 + self.ProblemData.mode) * (thetaIntegral + self.ProblemData.initial[-1] * (quad_mesh_2[j-i] - self.NumericalParameters.mesh[i])) ** 2 * mu_vals[j]
            eta[i] = simpson(integrand_1, x=quad_mesh_1) + simpson(integrand_2, x=quad_mesh_2)
        energy += eta
        energy_correct += 0.5 * eta
        return energy, energy_correct
    
    def precompute_integrals(self):
        """
        Precomputes integrals of the kernel.
        """
        mesh = np.linspace(0, self.NumericalParameters.final_time, self.NumericalParameters.n_elements)
        self.kernel_integrals = np.zeros(self.NumericalParameters.n_elements)
        for i in range(self.NumericalParameters.n_elements):
            # self.kernel_integrals[i], _ = quad(lambda t: self.kernel(t, np.array([0])), mesh[i], 25)
            self.kernel_integrals[i] = 2 ** (-np.ceil(mesh[i])) * (1.5 + (np.ceil(mesh[i]) - mesh[i]) * (np.ceil(mesh[i]) * 0.5 + 1 - mesh[i] * 0.5))
        return
    
    def exact(self, t: np.ndarray) -> np.ndarray:
        """
        Computes exact solution.
        """
        y1 = (1 / ((2 * np.pi) ** 2 + self.ProblemData.gamma ** 2 / self.ProblemData.beta ** 2)) * (self.ProblemData.gamma / self.ProblemData.beta * np.sin(2 * np.pi * t) - 2 * np.pi * np.cos(2 * np.pi * t))
        y2 = (1 / ((2 * np.pi) ** 2 + self.ProblemData.gamma ** 2 / self.ProblemData.beta ** 2)) * (self.ProblemData.gamma / self.ProblemData.beta * 2 * np.pi * np.cos(2 * np.pi * t) + 4 * np.pi ** 2 * np.sin(2 * np.pi * t))
        y3 = (1 / ((2 * np.pi) ** 2 + self.ProblemData.gamma ** 2 / self.ProblemData.beta ** 2)) * (- self.ProblemData.gamma / self.ProblemData.beta * 4 * np.pi ** 2 * np.sin(2 * np.pi * t) + 8 * np.pi ** 3 * np.cos(2 * np.pi * t))
        y4 = ((self.ProblemData.beta * self.ProblemData.mode - 4 * np.pi ** 2) / (self.ProblemData.rho * self.ProblemData.mode)) * np.sin(2 * np.pi * t)
        return [y1, y2, y3, y4]

def main():
    # === Instantiate solver ===
    omega = 2 * np.pi
    params = {'alpha': 1.0,
              'beta': 30.0,
              'gamma': 60.0,
              'delta': 1.0,
              'rho': 1.0,
              'mode': 1.0}
    initial_conditions = [- omega / (omega ** 2 + params['gamma'] ** 2 / params['beta'] ** 2), 
                          1 / (omega ** 2 + params['gamma'] ** 2 / params['beta'] ** 2) * omega * params['gamma'] / params['beta'], 
                          omega ** 3 / (omega ** 2 + params['gamma'] ** 2 / params['beta'] ** 2), 
                          0.0]
    solver = MGTGPSolver(initial_conditions, params)
    
    # l = 1
    mode = np.max([omega ** 2 * ((solver.ProblemData.beta + 1) - np.sqrt((solver.ProblemData.beta - 1) ** 2 + 4 * omega ** 2 * solver.ProblemData.rho ** 2)) / (2 * (solver.ProblemData.beta - omega ** 2 * solver.ProblemData.rho ** 2)), 
                   omega ** 2 * ((solver.ProblemData.beta + 1) + np.sqrt((solver.ProblemData.beta - 1) ** 2 + 4 * omega ** 2 * solver.ProblemData.rho ** 2)) / (2 * (solver.ProblemData.beta - omega ** 2 * solver.ProblemData.rho ** 2))])
    print(f'Setting mode = {mode}')
    
    solver.ProblemData.mode = mode
    solver.ProblemData.delta = 1 / mode
    print(f'delta = {1 / mode}')
    stability = solver.ProblemData.beta - solver.ProblemData.gamma / (solver.ProblemData.alpha + solver.ProblemData.delta * solver.ProblemData.mode)
    print(f'Stability = {stability}')

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
    solver.Settings.path_prefix = '../resonant/'
    solver.Settings.dump_label = 'resonant'

    # === Solve ===
    solver.precompute_integrals()
    solver.solve()

    # === Plot ===
    solver.plot(save_fig=True, show_errors=True, show_exact=True, fig_size=(3.2, 3.2), names=['u', 'u_t', 'u_{tt}', '\\theta'])

if __name__ == "__main__":
    main()