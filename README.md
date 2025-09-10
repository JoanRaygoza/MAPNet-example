# MAPNet
Contact: jraygoza@cicese.edu.mx

**Magnetic-Atmospheric Parameters Neural Network**

MAPNet is a neural‑network model that generate high‑resolution Stokes profiles (I, Q, U and V) for **5208 Å, 5171 Å and MZS**.

Given 13 magnetic and atmospheric parameters—phase, effective temperature, surface gravity, global metallicity, inclination angle, alpha, beta, gamma, y2, y3, dipolar moment, projected rotational velocity, and microturbulent velocity—the network outputs a synthetic flux vector with 96 wavelength points. Four model sizes let you trade accuracy for speed and memory.

The core library is light‑weight; the first time you instantiate a model, its weights are fetched from **Google Drive** via `gdown` and cached locally under `~/MAPNet_models/`.

---

## Astrophysical parameter space

| Parameter                  | Min   | Max   |
| -------------------------- | ----- | ----- |
| **phase** (period)         | 0.0   | 1.0   |
| **teff** (K)               | 7 000 | 7 500 |
| **log g** (dex)            | 4.0   | 4.5   |
| **\[M/H]** (dex)           | –2.0  | 0.0   |
| **inclination angle** (deg)| 0.0   | 180   |
| **alpha** (deg)            | -180  | 180   |
| **beta** (deg)             | 0.0   | 180   |
| **gamma** (deg)            | -180  | 180   |
| **y2** (r*)                | 0.0   | 0.2   |
| **y3** (r*)                | 0.0   | 0.2   |
| **dipolar moment** (gauss) | 100   | 4 500 |
| **v sin i** (km s⁻¹)       | 2.0   | 10.0  |
| **vmic** (km s⁻¹)          | 0.0   | 3.0   |

---

## Model variants
The model has 102M of parameters and the weights has a size of 1.23GB.
\*Sizes are approximate .h5 files downloaded on demand.

---

## Installation

> MAPNET is currently in private beta.

---

## Quick start 
### Synthetize — Single input
```python
import numpy as np
import mapnet

net = MAPNet(line="5208")
# phase, teff, logg, mh, incl, alpha, beta, gamma, y2, y3, m, vsini, vmic
x = np.array([0.17, 7157, 4.12, -1.6, 33, -11, 78, 3, 0.1, 0.1, 1300, 3.4, 0.5])

spectrum = net.synthetize_spectra(x)
print(spectrum.shape) # (1, 4, 96)
```

### Synthetize — Multiple inputs

```python
X = np.array([
    [0.1, 7100, 4.1, -2, 0, 15, 10, 3, 0.02, 0.1, 3000, 2, 0],
    [0.5, 7150, 4.0, 0, 90, 30, 0, 5, 0.1, 0.12, 1000, 2.9, 1.2],
    [0.7, 7300, 4.7, -1, 120, 0, 60, 0, 0.08, 0.0, 100, 8.6, 3.0],
])

synth = net.synthetize_spectra(X)
print(synth.shape)  # (3, 4, 96)
```

### Inversion with multiple phases - Estimate parameters from a spectrum
MAPNet can also be used to invert spectra, estimating the 13 magnetic-atmospheric parameters (Teff, logg, [M/H], Inclination Angle, Alpha, Beta, Gamma, y2, y3, m, vsini, vmic) from an observed flux vector in multiple phases. It uses Particle Swarm Optimization (PSO) to minimize the error between the observed spectrum and the network's prediction.

```python
solution, inv_spectra, fitness = net.inversion(
  spectrum_phases, 
  n_particles=1024, 
  iters=50, 
  verbose=1, 
  phases=phases)
```

This returns:
* **solution**: best-fit magnetic-atmospheric parameters found by the optimizer.
* **inv_spectra**: synthetic spectrum generated from the inferred parameters.
* **fitness**: final value of the objective function.

### Inversion with fixed parameters
You can fix specific magnetic-atmospheric parameters during the inversion by using the corresponding keyword arguments:
* `fixed_teff`
* `fixed_logg`
* `fixed_mh`
* `fixed_incl`
* `fixed_alpha`
* `fixed_beta`
* `fixed_gamma`
* `fixed_y2`
* `fixed_y3`
* `fixed_m`
* `fixed_vsini`
* `fixed_vmic`

For example, the following call fixes `teff` and `logg`:
```python
solution, inv_spectra, fitness = net.inversion(
  spectrum_phases, 
  n_particles=1024, 
  iters=50, 
  verbose=1, 
  phases=phases, 
  fixed_teff=7150, 
  fixed_logg=4.1)
```

### Inversion with parameter ranges
You can constrain parameters to a specific range using the following arguments:
* `teff_range`
* `logg_range`
* `mh_range`
* `incl_range`
* `alpha_range`
* `beta_range`
* `gamma_range`
* `y2_range`
* `y3_range`
* `m_range`
* `vsini_range`
* `vmic_range`

```python
solution, inv_spectra, fitness = net.inversion(
  spectrum_phases, 
  n_particles=1024, 
  iters=50, 
  verbose=1, 
  phases=phases, 
  teff_range=(7100,7200), 
  m_range=(1300,1700))
```
This limits `Teff` and `m` to specific ranges while leaving the other parameters free. You may also combine fixed values for some parameters with range limits for others.

### Inversion with selected Stokes
You can constrain the Stokes profiles to use in inversion method:

```python
solution, inv_spectra, fitness = net.inversion(
  spectrum_phases, 
  n_particles=1024, 
  iters=50, 
  verbose=1, 
  phases=phases, 
  use_Stokes_I=True, 
  use_Stokes_Q=False, 
  use_Stokes_U=False, 
  use_Stokes_V=True)
```

### Custom objective function
You can provide your own objective function to compare the observed and predicted spectra. It must accept two arguments: y_obs and y_pred.

Example using WMAPE with weight of Median WMAPE of each Stoke profile in line 5208:

```python
def obj(y_true, y_pred):
    err_stokes_i = 0
    err_stokes_q = 0
    err_stokes_u = 0
    err_stokes_v = 0
    for i in range(7):
        start_ph = i*384
        wmape_stokes_i = weighted_mean_absolute_percentage_error(y_true[:,start_ph:start_ph+96], y_pred[:,start_ph:start_ph+96])
        wmape_stokes_q = weighted_mean_absolute_percentage_error(y_true[:,start_ph+96:start_ph+192], y_pred[:,start_ph+96:start_ph+192])
        wmape_stokes_u = weighted_mean_absolute_percentage_error(y_true[:,start_ph+192:start_ph+288], y_pred[:,start_ph+192:start_ph+288])
        wmape_stokes_v = weighted_mean_absolute_percentage_error(y_true[:,start_ph+288:start_ph+384], y_pred[:,start_ph+288:start_ph+384])
        err_stokes_i += wmape_stokes_i/0.096
        err_stokes_q += wmape_stokes_q/8.68
        err_stokes_u += wmape_stokes_u/12.68
        err_stokes_v += wmape_stokes_v/3.63

    return (err_stokes_i+err_stokes_q+err_stokes_u+err_stokes_v)/28
```
Then use it like this:
```python
solution, inv_spectra, fitness = net.inversion(
  spectrum_phases, 
  n_particles=2048, 
  iters=50, 
  verbose=1, 
  phases=phases, 
  objective_function=obj)
```

---


### Important
Be sure to select the corresponding device for run the model, whether GPU or CPU.
If using the CPU, you can select the number of jobs to use. This configuration must be done before instantiating MAPNet.

```python
from mapnet import MAPNet, config
config(jobs=6)
net = MAPNet(line="5208")
...
```

## Extra utilities

```python
wl   = net.get_wavelength()  # ndarray of 96 wavelengths (Å)
```

---

## API (`mapnet`)
### `class MAPNet(line: str = "5171", batch_size: int = 512)`

Main interface for spectral synthesis and inversion using neural network models.

Model weights are downloaded on first use and cached locally in `~/MAPNet_models/`.
`batch_size` argument controls the amount of data to load on the model when it is predicting (synthetize and inversion).

---

### **Methods**

```python
synthetize_spectra(data: np.ndarray) -> np.ndarray
```

Generates synthetic spectra from stellar parameters.

* `data`: 1D or 2D NumPy array of shape `(13,)` or `(n_samples, 13)` in the order
  *(Phase, Teff, log g, \[M/H], Inclination Angle, Alpha, Beta, Gamma, y2, y3, Dipolar Moment, vsini, vmic)*.

---

```python
get_wavelength() -> np.ndarray
```

Returns the wavelength grid (shape: `(96,)`) used by the model.

---

```python
inversion(
  y_obs: np.ndarray, 
  n_particles: int = 2048, 
  iters: int = 50, 
  objective_function: Callable = default_objective, 
  W: float = 0.7,
  C1: float = 1.0,
  C2: float = 1.0, 
  fixed_phase: float | None = None, 
  fixed_teff: float | None = None, 
  fixed_logg: float | None = None, 
  fixed_mh: float | None = None, 
  fixed_incl: float | None = None, 
  fixed_alpha: float | None = None, 
  fixed_beta: float | None = None, 
  fixed_gamma: float | None = None, 
  fixed_y2: float | None = None, 
  fixed_y3: float | None = None, 
  fixed_m: float | None = None, 
  fixed_vsini: float | None = None, 
  fixed_vmic: float | None = None, 
  phase_range: tuple[float, float] = (0,1), 
  teff_range: tuple[float, float] = (7000,7500), 
  logg_range: tuple[float, float] = (4.0, 4.5), 
  mh_range: tuple[float, float] = (-2.0, 0.0), 
  incl_range: tuple[float, float] = (0, 180), 
  alpha_range: tuple[float, float] = (-180, 180), 
  beta_range: tuple[float, float] = (0, 180), 
  gamma_range: tuple[float, float] = (-180, 180), 
  y2_range: tuple[float, float] = (0, 0.2), 
  y3_range: tuple[float, float] = (0, 0.2), 
  m_range: tuple[float, float] = (100, 4500), 
  vsini_range: tuple[float, float] = (2.0, 10.0), 
  vmic_range: tuple[float, float] = (0.0, 3.0), 
  verbose: int = 0,
  phases: list | None = None, 
  use_Stokes_I: bool = True, 
  use_Stokes_Q: bool = True,
  use_Stokes_U: bool = True, 
  use_Stokes_V: bool = True) -> tuple[np.ndarray, np.ndarray, float]
```

Performs a global optimization using Particle Swarm Optimization (PSO) to infer the magnetic-atmospheric parameters that best reproduce the observed spectrum `y_obs`.
You can:
* Fix any subset of parameters using the `fixed_*` arguments.
* Restrict the search space of free parameters using the corresponding `*_range` tuples.
* Contraint which Stokes profiles to use in inversion using the corresponding `use_Stokes_*` arguments.
* Combine as needed.
**Parameters**:
* y_obs (`np.ndarray`): The observed flux vector to be fitted.
* n_particles (`int`): Number of particles used in the PSO swarm.
* iters (`int`): Number of optimization iterations.
* objective_function (`Callable`, optional): Objective function to minimize. Defaults to `default_objective`.
* W, C1, C2 (`float`): PSO hyperparameters controlling inertia and learning factors.
* fixed_* (`float | None`): Values to fix specific parameters during the inversion. If `None`, the parameter is optimized.
* *_range (`tuple[float, float]`): Search intervals for each parameter (used if not fixed).
* verbose (`int`): Verbosity level (0 = silent, 1 = progress info).
* use_Stokes_* (`bool`): Select Stoke profile to use in inversion.
* phases (`list`): fixed multiple phases for the inversion. 
**Returns**:
* `solution` (`np.ndarray`): Best-fit parameter vector.
* `inv_spectra` (`np.ndarray`): Synthetic spectrum corresponding to the best solution.
* `fitness` (`float`): Final error value of the best-fit solution.

---

## Troubleshooting

| Symptom                                          | Fix                                                                                                                 |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| *`ModuleNotFoundError: No module named 'gdown'`* | `pip install gdown`                                                                                                 |
| Slow download / blocked                          | Download the `.h5` manually from the Google Drive link in `_MODEL_TABLE` and place it under `~/MAPNet_models/`. |

---

## Contributing

1. Fork this repository and create a feature or bug‑fix branch.
2. Run the unit tests (`pytest`).
3. Open a pull request.

Bug reports and feature requests are welcome on the *Issues* page.

---

## License


MIT © 2025 Joan Raygoza
