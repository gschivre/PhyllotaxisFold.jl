# Notation and references

## Notation

| symbol | meaning |
|---|---|
| ``p``, ``n``, ``j`` | height on the stem (bottom-up), number of organs, site of the spiral (canonical order) |
| ``\sigma``, ``d_p = \sigma(p) - p``, ``N_d`` | order, displacement (``d < 0``: late organ), number of organs displaced by ``d`` |
| ``y_p``, ``x_p`` | absolute azimuth, stem length below position ``p`` |
| ``\phi``, ``o``, ``\psi``, ``\alpha`` | spiral offset, orientation, divergence (golden or Lucas), probability of golden |
| ``\kappa``, ``\mu_\kappa``, ``\sigma_\kappa`` | angular concentration of a stem, its log-normal population law |
| ``\omega_s``, ``m_\omega``, ``s_\omega`` | twist rate of stem ``s``, its population mean and sd |
| ``\eta_d`` | cost (log-propensity) of displacement ``d`` |
| ``A``, ``P_A(\sigma)`` | the propensity model ([§4](propensity.md)), its law of the order |
| ``P_f(\sigma)`` | the law of the order when the sites commit with independent delays ``f`` ([§5.2](timing.md)) |
| ``W``, ``\rho_{p,d}`` | weight matrix, marginal probability ``P(\sigma(p) = p + d)`` |
| ``X``, ``\mu`` | switch variable and margin above the fold, in noise units ([§5.1](timing.md)) |
| ``\tau_i``, ``e_i``, ``f`` | commitment time and delay of site ``i`` (plastochrons), delay law |
| ``s``, ``\pi_L``, ``m_L``, ``\pi_E``, ``m_E`` | core sd; fraction and mean extra delay of late sites; of early sites |
| ``\Delta\ell``, ``\Delta\ell_s`` | ``\log p(y \mid f) - \log p(y \mid A)``, summed over stems; for stem ``s`` |
| ``t_p`` | commitment time of the organ at height ``p``, ``t_p = \tau_{\sigma(p)}`` |

## References

- Besnard F. et al. (2014) Cytokinin signalling inhibitory fields provide robustness to phyllotaxis. *Nature* 505,
  417–421.
- Landrein B. et al. (2013) Impaired cellulose synthase guidance leads to stem torsion and twists phyllotactic patterns
  in *Arabidopsis*. *Curr. Biol.* 23, 895–900.
- Refahi Y. et al. (2016) A stochastic multicellular model identifies biological watermarks from disorders in
  self-organized patterns of phyllotaxis. *eLife* 5, e14093.
- Reinhardt D. et al. (2003) Regulation of phyllotaxis by polar auxin transport. *Nature* 426, 255–260.
- Shi B. & Vernoux T. (2019) Patterning at the shoot apical meristem and phyllotaxis. *Curr. Top. Dev. Biol.* 131,
  81–107.
- Valiant L. G. (1979) The complexity of computing the permanent. *Theor. Comput. Sci.* 8, 189–201.
