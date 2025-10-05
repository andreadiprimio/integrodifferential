import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

class EnergyDecayAnalyzer:

    def __init__(self, energies: np.ndarray):
        """
        Constructor.
        """
        self.energies = energies
        pass

    def energy_plotter(self, save_fig: bool = False, fpath: str = './'):
        """
        Energy plotter. Modify as needed.
        """
        self.__energy_cleaner()
        T = 1000
        N = 2000
        plt.figure(figsize=(6.4, 4))
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
        else:
            plt.show()
        T = 50
        N = 1000
        plt.figure(figsize=(6.4,4))
        for i in range(3, len(self.energies), 2):
            plt.semilogy(np.linspace(0, T, N), self.energies[i], label=f'Mode {i+1}', markersize=2)
            plt.title('Energy over time (higher modes)')
            plt.xlabel('$t$')
            plt.ylabel('Energy (log scale)')
            plt.grid(True)
            plt.legend(loc='best', fontsize=7, ncol=3)
        if save_fig:
            plt.savefig(fpath + 'energy_higher_modes.pdf')
            plt.close()
        else:
            plt.show()
        return
    
    def energy_dumper(self, fname: str = './'):
        np.savetxt(fname, self.energies, header="Energy data")
        return
    
    def __energy_cleaner(self) -> np.ndarray:
        for i in range(len(self.energies)):
            self.energies[i] = np.where(self.energies[i] >= np.finfo(np.float64).eps, self.energies[i], np.nan)
        return
    
    def estimate_decay_rates(self) -> np.ndarray:
        self.__energy_cleaner()
        self.rates = np.zeros(len(self.energies))
        T = 1000
        N = 2000
        for i in range(len(self.energies)):
            nancheck = np.isnan(self.energies[i])
            j = np.where(nancheck == True)
            if i == 3:
                T = 50
                N = 1000
            if np.size(j) == 0:
                j = N
            else:
                j = j[0][0]
            time = np.linspace(0, T, N)
            time = time[0:j]
            print(self.energies[i][0:j])
            slope, intercept, r_value, p_value, std_err = linregress(time, np.log10(self.energies[i][0:j]))
            self.rates[i] = -slope
            #plt.semilogy(time, self.energies[i][0:j], label=f'Mode {i+1}', markersize=2)
            #plt.semilogy(time, 10 ** (intercept + slope * time), '--', label=f'Fit mode {i+1}', markersize=2)
            #plt.title('Energy over time (higher modes)')
            #plt.xlabel('$t$')
            #plt.ylabel('Energy (log scale)')
            #plt.grid(True)
            #plt.legend(loc='best', fontsize=7, ncol=3)
            #plt.show()
        print(self.rates)
        return self.rates