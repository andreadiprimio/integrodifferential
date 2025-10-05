import matplotlib.pyplot as plt
import numpy as np

class EnergyDecayAnalyzer:

    def __init__(self, energies: np.ndarray):
        """
        Constructor.
        """
        self.energies = energies
        return

    def energy_plotter(self, save_fig: bool = False, fpath: str = './'):
        """
        Energy plotter. Modify as needed.
        """
        self.__energy_cleaner()
        T = 1000
        N = 2000
        for i in range(3):
            plt.semilogy(np.linspace(0, T, N), self.energies[i], label=f'Mode {i+1}', markersize=2)
            plt.title('Energy over time (lower modes)')
            plt.xlabel('$t$')
            plt.ylabel('Energy (log scale)')
            plt.grid(True)
            plt.legend()
        if save_fig:
            plt.savefig(fpath + 'energy_lower_modes.pdf')
            plt.close()
        T = 50
        N = 1000
        for i in range(4, len(self.energies)):
            plt.semilogy(np.linspace(0, T, N), self.energies[i], label=f'Mode {i+1}', markersize=2)
            plt.title('Energy over time (higher modes)')
            plt.xlabel('$t$')
            plt.ylabel('Energy (log scale)')
            plt.grid(True)
            plt.legend()
        if save_fig:
            plt.savefig(fpath + 'energy_higher_modes.pdf')
            plt.close()
        else:
            plt.show()
        return
    
    def energy_dumper(self, fpath: str = './'):
        fname = fpath + f'energies.txt'
        np.savetxt(fname, self.energies, header=f"Energy data")
        return
    
    def __energy_cleaner(self) -> np.ndarray:
        for i in range(len(self.energies)):
            self.energies[i] = np.where(self.energies[i] >= np.finfo(np.float64).eps, self.energies[i], np.nan)
        return