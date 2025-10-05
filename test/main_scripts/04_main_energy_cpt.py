import numpy as np
from scipy.integrate import simpson, quad
import matplotlib.pyplot as plt
from tqdm.rich import tqdm
from integrodifferential import IntegroDifferentialProblem
from energy_analyzer import EnergyDecayAnalyzer

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
        diff = t - s
        val = 0.5 - diff + diff ** 2 * 0.5
        return np.where(diff <= 1, val, 0.0)

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
        return np.where(t <= 1, 1 - t, 0.0)
    
    def compute_energy(self) -> np.ndarray:
        """
        Computes energy relative to mode, assuming theta(s) = theta(0) for all s < 0.
        """
        stability = self.ProblemData.beta - self.ProblemData.gamma / (self.ProblemData.alpha + self.ProblemData.delta * self.ProblemData.mode)
        if np.abs(stability) < 1e-15:
            m = 1
        else:
            m = 0
        u = self.Output.solution[0]
        v = self.Output.solution[1]
        w = self.Output.solution[2]
        theta = self.Output.solution[3]
        energy = (1 + self.ProblemData.mode) * u ** 2 + (1 + self.ProblemData.mode) * v ** 2 + w ** 2 + theta ** 2
        energy_correct = 0.5 * (self.ProblemData.beta * (1 + self.ProblemData.mode) * (v + self.ProblemData.gamma / self.ProblemData.beta * u) ** 2 + (w + self.ProblemData.gamma / self.ProblemData.beta * v) ** 2 + self.ProblemData.gamma * (self.ProblemData.alpha + m) / self.ProblemData.beta ** 2 * (self.ProblemData.beta - self.ProblemData.gamma / (self.ProblemData.alpha + m)) * v ** 2 + self.ProblemData.delta * self.ProblemData.gamma / self.ProblemData.beta * (1 + self.ProblemData.mode) * v ** 2 + theta ** 2)
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
            # self.kernel_integrals[i], _ = quad(lambda t: self.kernel(t, np.array([0])), mesh[i], np.inf)
            self.kernel_integrals[i] = np.max([0.0, (1-mesh[i]) ** 3 / 6])
        return

def main():
    # === Instantiate solver ===
    params = {'alpha': 49.0,
              'beta': 50.0,
              'gamma': 2500.0,
              'delta': 1.0,
              'rho': 1.0,
              'mode': 1.0}
    initial_conditions = [0.1, 0.1, 0.1, 0.1]
    solver = MGTGPSolver(initial_conditions, params)

    # === Configure numerical parameters ===
    solver.NumericalParameters.n_elements = 2000
    solver.NumericalParameters.final_time = 1000
    solver.NumericalParameters.tol = 1e-8
    solver.NumericalParameters.max_iter = 300
    solver.NumericalParameters.smoothing = 0.9
    solver.NumericalParameters.solver = "Radau"

    # === Configure settings ===
    solver.Settings.dump_parameters = False
    solver.Settings.dump_solution = False
    solver.Settings.dump_convergence_data = False
    solver.Settings.path_prefix = '../decay_rate/'
    solver.Settings.dump_label = 'decay'

    # === Set up modes ===
    modes = np.arange(20, 100, 10)
    energies = np.array([])
    energies_correct = np.array([])

    # === Solve ===
    for mode in tqdm(modes):
        solver.ProblemData.mode = mode
        if mode >= 4:
            solver.NumericalParameters.n_elements = 500
            solver.NumericalParameters.final_time = 50
        
        solver.precompute_integrals()
        solver.solve()

        # === Plot ===
        # solver.plot(save_fig=True, show_errors=True, show_exact=True, names=['u', 'u_t', 'u_{tt}', '\\theta'])

        # === Compute energy ===
        val, val_correct = solver.compute_energy()
        energies = np.append(energies, val)
        energies_correct = np.append(energies_correct, val_correct)
    
    eda = EnergyDecayAnalyzer(energies)
    eda.energy_dumper(fpath=solver.Settings.path_prefix)
    eda.energy_plotter(save_fig=True, fpath=solver.Settings.path_prefix)
    
    eda = EnergyDecayAnalyzer(energies_correct)
    eda.energy_dumper(fpath=solver.Settings.path_prefix + 'energy_correct/')
    eda.energy_plotter(save_fig=True, fpath=solver.Settings.path_prefix + 'energy_correct/')

if __name__ == "__main__":
    main()