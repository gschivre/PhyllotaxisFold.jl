# Notation and references

## Notation

| symbol | meaning |
|---|---|
| ``p``, ``n``, ``j`` | height on the stem (bottom-up), number of organs, site of the spiral (canonical order) |
| ``s``, ``S``, ``n_s`` | stem, number of stems, number of organs of stem ``s``; the subscript ``s`` marks a quantity of stem ``s`` (``y_{sp}``, ``\kappa_s``) |
| ``\sigma``, ``d_p = \sigma(p) - p``, ``N_d`` | order, displacement (``d < 0``: late organ), number of organs displaced by ``d`` |
| ``y_p``, ``x_p`` | absolute azimuth, stem length below position ``p`` |
| ``\phi``, ``o``, ``\psi``, ``\alpha`` | spiral offset, orientation, divergence (golden or Lucas), probability of golden |
| ``\kappa``, ``\mu_\kappa``, ``\sigma_\kappa`` | angular concentration of a stem, its log-normal population law |
| ``\omega_s``, ``m_\omega``, ``s_\omega`` | twist rate of stem ``s``, its population mean and sd |
| ``\pi_o``, ``z_{sp}`` | probability of an outlier angle, whether angle ``p`` of stem ``s`` is one ([§2.4](angles.md#2.4-Outlier-angles)) |
| ``\eta_d`` | cost (log-propensity) of displacement ``d`` |
| ``\Delta_d`` | gap between successive costs, ``\eta_d = -(\Delta_1 + \dots + \Delta_d)`` ([§4.3](propensity.md#4.3-Prior)) |
| ``\mu_\Delta, \sigma_\Delta``, ``m_\kappa, s_\kappa``, ``\nu_\kappa, \tau_\kappa``, ``a_\alpha, b_\alpha``, ``a_\pi, b_\pi`` | fixed parameters of the priors on ``\Delta``, ``\log\mu_\kappa``, ``\sigma_\kappa``, ``\alpha``, ``\pi_o`` (`PhyllotaxisPrior`) |
| ``m_t, s_t``, ``\nu_t, \tau_t`` | fixed parameters of the priors on ``m_\omega``, ``s_\omega`` (`TorsionPrior`) |
| ``A``, ``P_A(\sigma)`` | the propensity model ([§4](propensity.md)), its law of the order |
| ``P_f(\sigma)`` | the law of the order when the sites commit with independent delays ``f`` ([§5.2](timing.md#5.2-The-law-of-the-order-of-a-paced-meristem)) |
| ``W``, ``\rho_{p,d}`` | weight matrix, marginal probability ``P(\sigma(p) = p + d)`` |
| ``X``, ``\mu`` | switch variable and margin above the fold, in noise units ([§5.1](timing.md#5.1-Why-a-random-delay)) |
| ``\tau_i``, ``e_i``, ``f`` | commitment time and delay of site ``i`` (plastochrons), delay law |
| ``\sigma_c``, ``\pi_L``, ``m_L``, ``\pi_E``, ``m_E`` | core sd; fraction and mean extra delay of late sites; of early sites |
| ``q_d`` | overtaking profile: ``P(\text{site } i + d \text{ commits before site } i)`` ([§5.6](timing.md#5.6-The-delay-law)) |
| ``\Delta\ell``, ``\Delta\ell_s`` | ``\log p(y \mid f) - \log p(y \mid A)``, summed over stems; for stem ``s`` |
| ``t_p`` | commitment time of the organ at height ``p``, ``t_p = \tau_{\sigma(p)}`` |

## References

```@bibliography
```
