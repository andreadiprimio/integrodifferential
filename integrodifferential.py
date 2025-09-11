import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, make_dataclass, asdict
from abc import ABC, abstractmethod
from collections.abc import Callable
import matplotlib.pyplot as plt

@dataclass
class NumericalParameters:
    """
    Helper class to handle numerical parameters.
    """
    n_elements: int = 500
    tol: float = 1e-8
    smoothing: float = 0.5
    max_iter: int = 300
    solver: str = 'RK45'
    final_time: float = 50
    exact: Callable[[np.ndarray], np.ndarray] = None
    mesh: np.ndarray = None

    def __str__(self):
        """
        Prints class in format tag: value.
        """
        attrs = {**type(self).__dict__, **self.__dict__}
        attrs = {
            k: v for k, v in attrs.items()
            if not k.startswith("__") and not callable(v)
        }
        return "\n".join(f"{k}: {v}" for k, v in attrs.items())

@dataclass
class Output:
    """
    Helper class to handle outputs.
    """
    solution: np.ndarray = None
    errors: np.ndarray = None

@dataclass
class Settings:
    """
    Helper class to handle settings.
    """
    dump_solution: bool = True
    dump_parameters: bool = True
    dump_convergence_data: bool = False
    dump_label: str = 'test'
    path_prefix: str = './'

class IntegroDifferentialProblem(ABC):
    """
    Abstract base class for a generic linear integro-differential first-order system.
    """

    @abstractmethod
    def source(self, t: float) -> np.ndarray:
        """
        The function g such that y' = F(t, y) + memory(t, y) + g(t).
        """
        pass

    @abstractmethod
    def memoryless_operator(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        The operator F such that y' = F(t, y) + memory(t, y) + g(t).
        """
        pass

    @abstractmethod
    def kernel(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Interaction kernel to compute memory term. Expected integration with respect to s.
        """
        pass

    @abstractmethod
    def compute_memory(self, y: np.ndarray) -> np.ndarray:
        """
        Function to compute the memory term.
        """
        pass

    @abstractmethod
    def similarity_metric(self, guess: np.ndarray, solution: np.ndarray) -> float:
        """
        Function to compute the distance between old and new guesses.
        """
        pass

    def __dump(self):
        """
        Dumps results to text files.
        """
        if self.Settings.dump_convergence_data:
            filePath = self.Settings.path_prefix + 'convergence_' + self.Settings.dump_label + '.txt' 
            np.savetxt(filePath, self.Output.errors, header="Iteration errors")
        if self.Settings.dump_parameters:
            filePath = self.Settings.path_prefix + 'problem_parameters_' + self.Settings.dump_label + '.txt'
            params = asdict(self.ProblemData)
            with open(filePath, 'w') as file:
                for k in params.keys():
                    file.write(str(k)+': '+str(params[k])+'\n')
                file.close()
            filePath = self.Settings.path_prefix + 'numerical_parameters_' + self.Settings.dump_label + '.txt'
            params = asdict(self.NumericalParameters)
            with open(filePath, 'w') as file:
                for k in list(params.keys())[:-2]:
                    file.write(str(k)+': '+str(params[k])+'\n')
                file.close()
        if self.Settings.dump_solution:
            filePath = self.Settings.path_prefix + 'solution_' + self.Settings.dump_label + '.txt'
            np.savetxt(filePath, self.Output.solution, header="Solution data")
        return
    
    def plot(self, save_fig: bool = False, flabel: str = 'solution_plot', show_exact: bool = False, show_errors: bool = False, fig_size: tuple[float, float] = (3.5, 3.5), names: list[str] = None):
        """
        Plots solution.
        """
        invalid = "\\/"
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.figsize'] = fig_size[0], fig_size[1]
        plt.rcParams["savefig.bbox"] = 'tight'
        plt.rcParams["savefig.dpi"] = 'figure'
        plt.rcParams["savefig.format"] = 'pdf'
        if names is None:
            names = [f'y_{{{i+1}}}' for i in range(len(self.Output.solution))]
        if show_exact:
            exact = self.NumericalParameters.exact(self.NumericalParameters.mesh)
        for i in range(len(self.Output.solution)):
            plt.plot(self.NumericalParameters.mesh, self.Output.solution[i], label='Solution plot ($'+names[i]+'$)', color='blue')
            if show_exact:
                plt.plot(self.NumericalParameters.mesh, exact[i], '-.', label='Exact solution', color='red')
                plt.title('Approximate solution vs exact solution ($'+names[i]+'$)')
            else:
                plt.title('Approximate solution ($'+names[i]+'$)')
            plt.xlabel('$t$')
            plt.ylabel('$'+names[i]+'$')
            plt.grid(True)
            plt.legend()
            if save_fig:
                plt.savefig(self.Settings.path_prefix + flabel + f'_{names[i].strip(invalid)}.pdf')
                plt.close()
            else:
                plt.show()
            if show_errors:
                plt.semilogy(self.NumericalParameters.mesh, np.abs(self.Output.solution[i] - exact[i]), color = 'blue')
                plt.title('Error over time ($'+names[i]+'$)')
                plt.xlabel('$t$')
                plt.ylabel(f'$|{names[i]}^{{exact}} - {names[i]}|$')
                plt.grid(True)
                if save_fig:
                    plt.savefig(self.Settings.path_prefix + 'error' + f'_{names[i].strip(invalid)}.pdf')
                    plt.close()
                else:
                    plt.show()
        return
    
    def __compute_right_hand_side(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Internal routine to estimate y' without memory as F(t, y) + g(t).
        """
        return self.memoryless_operator(t, y) + self.source(t)
    
    def __set_memory_operator(self) -> np.ndarray:
        """
        Builds the complete integro-differential operator. It assumes the equation with memory is the last one.
        """
        def memory_operator(t: float, y: np.ndarray) -> np.ndarray:
            T = self.NumericalParameters.final_time
            N = self.NumericalParameters.n_elements
            memory_interp = lambda t_val: self.memory[min(int(np.floor(np.clip(t_val, 0, T) * (N - 1) / T)), N - 1)]
            return self.__compute_right_hand_side(t, y) + np.array([0, 0, 0, memory_interp(t)])
        return memory_operator
    
    def __set_problem_data(self, initial: np.ndarray, **params):
        """
        Creates a data class with problem-specific parameters.
        """
        fields = [(k, type(v)) for k, v in params.items()] + [('initial', list[float])]
        ProblemData = make_dataclass('ProblemData', fields)
        params['initial'] = initial
        self.ProblemData = ProblemData(**params)
        return
          
    def __init__(self, initial, **params):
        """
        Forces all derived classes to build data classes for parameters.
        """
        self.__set_problem_data(initial, **params)
        self.NumericalParameters = NumericalParameters()
        self.Output = Output() 
        self.Settings = Settings()
        return
    
    def solve(self):
        """Solve routine."""
        self.memory = np.zeros(self.NumericalParameters.n_elements)
        self.NumericalParameters.mesh = np.linspace(0, self.NumericalParameters.final_time, self.NumericalParameters.n_elements)
        iter = 0
        guess = solve_ivp(self.__compute_right_hand_side,
                        [0, self.NumericalParameters.final_time],
                        self.ProblemData.initial,
                        method=self.NumericalParameters.solver,
                        t_eval=self.NumericalParameters.mesh,
                        rtol=1e-8,
                        atol=1e-10).y
        done = False
        iteration_errors = []
        while not done and iter < self.NumericalParameters.max_iter:
            iter += 1
            self.compute_memory(guess[-1])
            memory_operator = self.__set_memory_operator()
            vector_solution = solve_ivp(memory_operator,
                                    [0, self.NumericalParameters.final_time],
                                    self.ProblemData.initial,
                                    method=self.NumericalParameters.solver,
                                    t_eval=self.NumericalParameters.mesh,
                                    rtol=1e-8,
                                    atol=1e-10).y
            current_error = self.similarity_metric(guess, vector_solution)
            print(f'current_error = {current_error}') # temporary
            iteration_errors.append(current_error)
            if current_error <= self.NumericalParameters.tol:
                done = True
                print(f"Converged after {iter} iterations with error {current_error:.2e}")
            else:
                guess = self.NumericalParameters.smoothing * vector_solution + (1 - self.NumericalParameters.smoothing) * guess
        if iter >= self.NumericalParameters.max_iter:
            print(f"Warning: Maximum iterations ({self.NumericalParameters.max_iter}) reached without convergence.")
            print(f"Final error: {current_error[-1]:.2e}")
        self.Output.solution = guess
        self.Output.errors = iteration_errors
        self.__dump()
        return