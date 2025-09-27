# integrodifferential: a Python solver for integro-differential equations
A Python-based solver for quasilinear ordinary integro-differential equations.
## What is the purpose of this solver?
The aim of this project is to provide a user-friendly tool to numerically solve ordinary differential equations with memory taking the form

$$u^{(n)} + f(t, u, u', ..., u^{(n-1)}) + \int_0^{+\infty} g(t,s)F(u(s)) \mathrm{d} s = 0$$

endowed with suitable initial conditions. Assuming to know all necessary input data, the code workflow is briefly described in the following diagram. 
```mermaid
flowchart LR
    A["Solve the equation without integral term"] --> B[""Estimate memory term using the computed solution""]
    B --> C["Solve the equation using the computed memory term as forcing"]
    C --> D{"The algorithm has reached convergence?"}
    D -- Yes --- E["Output the new solution"]
    D -- No --- F["Update guessed solution"]
    F --> B
```
Once a solution is obtained, a plotting method is already set up for visualization, possibly also displaying a known exact solution for a direct comparison.
## Requirements and dependencies
The following requirements are necessary to run the solver:
- Python 3.12.3 or higher
- `numpy` version 2.2.3 or higher
- `scipy` version 1.16.0 or higher
- `matplotlib` version 3.10.3 or higher

The code can run on any OS that has Python installed.
> [!NOTE]
> The version references may not be mandatory, but are recommended since they match the dev testing environment.
## Installation and usage
1. Clone the GitHub repository.
2. Fill the script `main.py` following the instructions therein.
3. Run `main.py`.
## Test cases
The folder `test` contains several examples of ready-to-run scripts analyzing the Moore-Gibson-Thompson equation.
## References
[TODO]
## Authors
Andrea Di Primio (andrea.diprimio@polimi.it)  
Lorenzo Liverani (lorenzo.liverani@fau.de)

