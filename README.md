# MAPNet

MAPNet is a neural-network package for synthesizing four Stokes profiles (I, Q, U, V) and estimating stellar parameters from observed spectra. The package currently provides the `4982` and `MZS_500_510` model lines.

The first model load downloads its weights from Google Drive and caches them in the `MAPNet_models/` directory in the current working directory. TensorFlow and the model dependencies must be installed in the Python environment.

## Installation

Install the package and its dependencies in your environment. This project is currently distributed in private beta; installation instructions will be provided with the release.

## Quick start

### Load a model

```python
from mapnet import MAPNet

net = MAPNet(line="4982")
```

Available line identifiers are `"4982"` and `"MZS_500_510"`. The model uses 96 wavelength points. Retrieve its wavelength grid with:

```python
wavelength = net.get_wavelength()
```

### Synthesize spectra

Each input row contains 12 physical parameters in this order: phase, effective temperature (`teff`), surface gravity (`logg`), metallicity (`mh`), inclination angle (`incl`), `alpha`, `beta`, `gamma`, `y2`, `y3`, dipolar magnetic moment (`m`), and projected rotation speed (`vsini`).

```python
import numpy as np

parameters = np.array([
    0.17, 7157, 4.12, -1.6, 33, -11, 78, 3, 0.1, 0.1, 1300, 3.4
])

spectra = net.synthetize_spectra(parameters)
print(spectra.shape)  # (1, 4, 96)
```

For multiple parameter sets, pass a two-dimensional array with shape `(n_models, 12)`. The result has shape `(n_models, 4, 96)`.

### Invert observed spectra

`inversion` fits multiple observations using particle swarm optimization. The input must have shape `(n_observations, 4, n_wavelength_points)` and include the Stokes profiles in I, Q, U, V order.

```python
# observed_spectra has shape (n_observations, 4, 96)
phases = [0.1, 0.3, 0.6]

solution, fitted_spectra, fitness = net.inversion(
    observed_spectra,
    phases=phases,
    n_particles=1024,
    iters=50,
    verbose=1,
)
```

The method returns:

- `solution`: fitted physical parameters. The first 11 values are the base parameters (`teff`, `logg`, `mh`, `incl`, `alpha`, `beta`, `gamma`, `y2`, `y3`, `m`, and `vsini`). Optional group metallicities and offsets follow when enabled.
- `fitted_spectra`: synthesized spectra for the best solution, with the same shape and wavelength steps as the observations.
- `fitness`: best objective value found.

### Fix parameters and set search ranges

Pass a physical value to a `fixed_*` argument to hold that parameter constant. Use the corresponding `*_range` argument to set the inclusive lower and upper bounds for a free parameter.

```python
solution, fitted_spectra, fitness = net.inversionv(
    observed_spectra,
    phases=phases,
    fixed_teff=7150,
    fixed_logg=4.1,
    mh_range=(-1.5, 0.0),
    m_range=(1300, 1700),
)
```

Supported fixed parameters are `fixed_teff`, `fixed_logg`, `fixed_mh`, `fixed_incl`, `fixed_alpha`, `fixed_beta`, `fixed_gamma`, `fixed_y2`, `fixed_y3`, `fixed_m`, and `fixed_vsini`. Corresponding range parameters are `teff_range`, `logg_range`, `mh_range`, `incl_range`, `alpha_range`, `beta_range`, `gamma_range`, `y2_range`, `y3_range`, `m_range`, and `vsini_range`; phases can be fixed with `phases` or searched within `phase_range`.

### Select Stokes profiles

By default, all four profiles contribute to the fit. Disable any profile with its corresponding flag:

```python
solution, fitted_spectra, fitness = net.inversionv(
    observed_spectra,
    phases=phases,
    use_Stokes_I=True,
    use_Stokes_Q=False,
    use_Stokes_U=False,
    use_Stokes_V=True,
)
```

### Group-specific metallicity and spectral offsets

Set `mh_phase=True` to estimate a separate metallicity for each observation. By default, observations with the same phase share a metallicity.

Pass one `(min_offset, max_offset)` interval per observation with `offsets` to fit spectral shifts. The fitted offsets are appended after the metallicity values in `solution`. The printed summary reports metallicity by observation and offsets by observation when these options are enabled.

```python
solution, fitted_spectra, fitness = net.inversion(
    observed_spectra,
    phases=phases,
    mh_phase=True,
    offsets=[(-0.02, 0.02)] * len(phases),
)
```

### Custom wavelength grids, instrumental response, and weights

- `wavelength_obs`: observed wavelength array with shape `(n_observations, n_wavelength_points)`. Defaults to the model grid repeated for each observation.
- `LSF_function`: callable receiving `(wavelength_syn, spectra)` and returning spectra convolved with the instrument line-spread function.
- `stokes_weights`: non-negative array with shape `(n_observations, 4)`. Positive weights cannot be assigned to disabled Stokes profiles.
- `objective_function`: objective factory. By default, `weighted_mse_factory` computes the mean squared error, optionally weighted by `stokes_weights`.

## TensorFlow CPU configuration

Call `config` before importing TensorFlow to configure TensorFlow's thread count:

```python
from mapnet import config

config(jobs=6)

from mapnet import MAPNet
net = MAPNet(line="4982")
```

## API overview

- `MAPNet(line="4982", batch_size=512)`: load a supported model and its input/output scalers.
- `MAPNet.synthetize_spectra(data)`: synthesize Stokes profiles from one or more 12-value physical parameter vectors.
- `MAPNet.inversion(y_obs, ...)`: fit parameters to multiple observed spectra. Supports fixed parameters, ranges, selected Stokes profiles, variable metallicity by phase, wavelength offsets, instrumental response, and Stokes weights.
- `MAPNet.get_wavelength()`: return the model wavelength grid.

## Notes

- The supported model lines and parameter count above reflect the implementation in this repository. Model weights are downloaded on first use and require network access.
- `inversion` defaults to 2048 particles, 50 iterations, and a local-neighborhood half-width (`neighboors`) of 32. Larger searches may improve exploration but take longer.
- `MAPNet` selects an available GPU when TensorFlow detects one; otherwise, it reports CPU use.

## License

See [LICENSE](./LICENSE) for license details.
