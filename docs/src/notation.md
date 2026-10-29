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
| ``\pi_o``, ``z_{sp}`` | probability of an outlier angle, whether angle ``p`` of stem ``s`` is one ([§2.4](angles.md)) |
| ``\eta_d`` | cost (log-propensity) of displacement ``d`` |
| ``\Delta_d`` | gap between successive costs, ``\eta_d = -(\Delta_1 + \dots + \Delta_d)`` ([§4.3](propensity.md)) |
| ``\mu_\Delta, \sigma_\Delta``, ``m_\kappa, s_\kappa``, ``\nu_\kappa, \tau_\kappa``, ``a_\alpha, b_\alpha``, ``a_\pi, b_\pi`` | fixed parameters of the priors on ``\Delta``, ``\log\mu_\kappa``, ``\sigma_\kappa``, ``\alpha``, ``\pi_o`` (`PhyllotaxisPrior`) |
| ``m_t, s_t``, ``\nu_t, \tau_t`` | fixed parameters of the priors on ``m_\omega``, ``s_\omega`` (`TorsionPrior`) |
| ``A``, ``P_A(\sigma)`` | the propensity model ([§4](propensity.md)), its law of the order |
| ``P_f(\sigma)`` | the law of the order when the sites commit with independent delays ``f`` ([§5.2](timing.md)) |
| ``W``, ``\rho_{p,d}`` | weight matrix, marginal probability ``P(\sigma(p) = p + d)`` |
| ``X``, ``\mu`` | switch variable and margin above the fold, in noise units ([§5.1](timing.md)) |
| ``\tau_i``, ``e_i``, ``f`` | commitment time and delay of site ``i`` (plastochrons), delay law |
| ``\sigma_c``, ``\pi_L``, ``m_L``, ``\pi_E``, ``m_E`` | core sd; fraction and mean extra delay of late sites; of early sites |
| ``q_d`` | overtaking profile: ``P(\text{site } i + d \text{ commits before site } i)`` ([§5.6](timing.md)) |
| ``\Delta\ell``, ``\Delta\ell_s`` | ``\log p(y \mid f) - \log p(y \mid A)``, summed over stems; for stem ``s`` |
| ``t_p`` | commitment time of the organ at height ``p``, ``t_p = \tau_{\sigma(p)}`` |

## References

- Alimohammadi Y. & Asgari K. (2025) Mallows model with learned distance metrics: sampling and maximum likelihood
  estimation. *arXiv* 2507.08108.
- Berglund N. & Gentz B. (2002) The effect of additive noise on dynamical hysteresis. *Nonlinearity* 15, 605–632.
- Besnard F. et al. (2014) Cytokinin signalling inhibitory fields provide robustness to phyllotaxis. *Nature* 505,
  417–421.
- Ferrell J. E. Jr. (2012) Bistability, bifurcations, and Waddington's epigenetic landscape. *Curr. Biol.* 22,
  R458–R466.
- Galvan-Ampudia C. S. et al. (2020) Temporal integration of auxin information for the regulation of patterning.
  *eLife* 9, e55832.
- Landrein B. et al. (2013) Impaired cellulose synthase guidance leads to stem torsion and twists phyllotactic patterns
  in *Arabidopsis*. *Curr. Biol.* 23, 895–900.
- Mallows C. L. (1957) Non-null ranking models. I. *Biometrika* 44, 114–130.
- Refahi Y. et al. (2016) A stochastic multicellular model identifies biological watermarks from disorders in
  self-organized patterns of phyllotaxis. *eLife* 5, e14093.
- Reinhardt D. et al. (2003) Regulation of phyllotaxis by polar auxin transport. *Nature* 426, 255–260.
- Sáez M., Briscoe J. & Rand D. A. (2022) Dynamical landscapes of cell fate decisions. *Interface Focus* 12,
  20220002.
- Scheffer M. et al. (2009) Early-warning signals for critical transitions. *Nature* 461, 53–59.
- Shi B. & Vernoux T. (2019) Patterning at the shoot apical meristem and phyllotaxis. *Curr. Top. Dev. Biol.* 131,
  81–107.
- Thurstone L. L. (1927) A law of comparative judgment. *Psychol. Rev.* 34, 273–286.
- Valiant L. G. (1979) The complexity of computing the permanent. *Theor. Comput. Sci.* 8, 189–201.
- Veraart A. J. et al. (2012) Recovery rates reflect distance to a tipping point in a living system. *Nature* 481,
  357–359.
- Wedgwood K. C. A. & Ashwin P. (2022) Morphogen-directed cell fate boundaries: slow passage through bifurcation and
  the role of folded saddles. *J. Theor. Biol.* 549, 111220.
