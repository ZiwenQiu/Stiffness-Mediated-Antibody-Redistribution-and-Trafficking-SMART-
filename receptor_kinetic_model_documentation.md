# Receptor Kinetic Model — Notebook Documentation

**File:** `receptor_kinetic_model_final.ipynb`  
**Manuscript:** Qiu et al., *Engineered stiffness-mediated antibody redistribution and trafficking (SMART) potentiates checkpoint immunotherapy in hepatocellular carcinoma*, 

---

## Experimental Design

### Scientific Question

How does nanoparticle mechanical stiffness (Young's modulus *E*) quantitatively govern FcγR surface availability (*F*) on Kupffer cells, and what stiffness threshold (E₅₀) must a nanoparticle exceed to reduce FcγR availability by ≥50%? This notebook derives a mechanistically grounded kinetic model, fits it to experimental data, and produces an actionable engineering design map.

### Mechanistic Derivation

The model is derived from a set of coupled ordinary differential equations describing FcγR trafficking at steady state. The key physical assumptions are:

1. **Stiffness-dependent internalisation rate.** Stiffer nanoparticles generate larger membrane contact areas (Hertz contact mechanics: $A_{\text{contact}} \propto (E^*)^{-2/3}$) and higher force-loading rates at the cell membrane. Based on the molecular clutch framework, the probability that a bound receptor complex enters the internalisation pathway before dissociating increases with loading rate. This is parametrised as a power law:

$$k_{\text{int}}^N(E) = k_0 \left(\frac{E}{E_0}\right)^\alpha$$

2. **Steady-state receptor balance.** At steady state, the rate of FcγR internalisation equals the rate of recycling from endosomes to the membrane. Solving the three coupled ODEs (for surface receptor Rₛ, endosomal receptor Rᵢ, and nanoparticle–receptor complex Bₙ) yields:

$$F(E) = \frac{R_s^*}{R_{\text{tot}}} = \frac{1}{1 + \Theta(E)}, \quad \Theta(E) = \frac{k_{\text{on}}^N c_N}{k_{\text{rec}}} \cdot \frac{k_{\text{int}}(E)}{k_{\text{off}} + k_{\text{int}}(E)}$$

3. **Two-parameter reduction.** When β = *k*_off / *k*₀ → 0 (internalisation rate dominates over dissociation within the experimental stiffness window), Θ(E) simplifies to A·(E/E₀)^α, yielding the two-parameter working model.

4. **Antibody uptake proportional to receptor availability.** Since αPD-L1 is captured via FcγR-mediated endocytosis:

$$U(E) = U_0 \cdot F(E)$$

### Dataset

| Platform | Conditions | E range (MPa) | Replicates (E) | Replicates (F, U) |
|----------|-----------|--------------|----------------|-------------------|
| β-glucan NPs | 10 (primary fit) | 4.10–191.84 | 60 AFM curves | 3 biological |
| NIPAM NPs | 5 (validation only) | 1.49–22.38 | 60 AFM curves | — (F only) |

---

## Software and Reproducibility

| Item | Specification |
|------|--------------|
| Language | Python ≥ 3.9 |
| Libraries | NumPy, SciPy, scikit-learn, Matplotlib |
| Random seed | 2025 (residual bootstrap) |
| Bootstrap resamples | B = 5,000 |
| Resampling strategy | Residual bootstrap (E positions fixed; residuals resampled) |

---

## Cell-by-Cell Annotation

### §0 — Environment Setup

**Purpose:** Import libraries, set constants, configure Matplotlib, define formatting helpers.

**Key decisions:**

- `SEED = 2025` and `N_BOOT = 5_000` are set as module-level constants. The residual bootstrap uses 5,000 resamples (vs. 20,000 in the statistical notebook) because each iteration requires a full nonlinear `curve_fit` optimisation, making it computationally more expensive than the OLS operations in the mediation notebook.
- `np.random.seed(SEED)` sets the global NumPy seed. The residual bootstrap additionally uses `np.random.default_rng(SEED)` for reproducible integer sampling.
- `fmt_p()` and `ci4()` are identical to those in the statistical analysis notebook, ensuring uniform formatting across both notebooks.

---

### §1 — Data Loading and Descriptive Statistics

**Purpose:** Read E, F, U data from Excel; compute condition-level means and standard deviations; set the fixed reference stiffness E₀_ref.

**E₀_ref derivation:**

```python
E0_ref = float(np.exp(np.mean(np.log(E_data))))
```

E₀_ref is set to the geometric mean of the β-glucan E values rather than 1.0 or a manually chosen reference. This log-scale centering minimises the numerical co-linearity between α and A during optimisation: when E is expressed relative to its geometric mean, the Jacobian matrix of the model is better conditioned, leading to more stable convergence and more reliable standard error estimates from the covariance matrix.

**NIPAM NP data:** These five conditions are hardcoded as numpy arrays rather than read from file, because the NIPAM F measurements were obtained from a separate experimental series and are used exclusively in §6 for cross-material validation. They do not enter the primary fitting procedure.

**parse_sheet():** Generalised function replacing the original separate functions for U and F; accepts column indices as arguments to handle both sheets with a single implementation.

**Output:** Summary table with CL index, E mean ± SD ± CV%, F mean ± SD, U mean ± SD for all 10 β-glucan conditions.

---

### §2 — Statistical Validation

**Purpose:** Confirm that stiffness has a statistically significant and largely monotone effect on both F and U before fitting a parametric model.

**One-way ANOVA** tests whether group means differ across the 10 crosslinking conditions under the assumption of normally distributed, equal-variance residuals. **Kruskal–Wallis** is the non-parametric equivalent and makes no distributional assumption. Both tests reject H₀ (all conditions equal) if at least one condition mean differs; they do not test monotonicity directly.

**Adjacent-condition t-tests** check pairwise monotonicity. The two-sided test is used because the biological hypothesis (more crosslinking → stiffer → more FcγR internalisation → lower F) is monotone decreasing, but outlier conditions (e.g., CL3→CL4 non-monotone transition noted in the original notebook) may arise from measurement noise. Non-significant non-monotone transitions are expected and do not invalidate the monotone model assumption.

**Output:** ANOVA and Kruskal–Wallis P-values for F and U; table of adjacent-condition t-tests with direction and significance.

---

### §3 — Model Definition

**Purpose:** Define the mathematical functions used throughout the notebook. All subsequent analysis depends on these three functions.

**`F_model(E, alpha, A)`** implements:

$$F(E) = \frac{1}{1 + A \cdot (E/E_{0,\text{ref}})^\alpha}$$

`E0_ref` is accessed from the module scope (not passed as argument) because it is a fixed constant, not a fitting parameter. This prevents accidental variation of E₀ during bootstrap iterations.

**`joint_model(E_double, alpha, A, U0)`** concatenates F and U predictions for simultaneous fitting:

```python
E_double = [E_for_F; E_for_U]   # length 2n = 20
output   = [F_pred;  U_pred]
```

Joint fitting is essential because *F* and *U* share the same (α, A) parameters but have different measurement uncertainties. Sequential fitting (first fit F, then use fitted α, A to scale U by U₀) would ignore the information content of U measurements in constraining α and A, and would typically over-inflate R²(U) by selecting α, A optimised for F rather than the joint optimum.

**`E50_calc(alpha, A)`** solves $F(E_{50}) = 0.5$ analytically:

$$1 + A(E_{50}/E_0)^\alpha = 2 \implies E_{50} = E_0 \cdot (1/A)^{1/\alpha}$$

**`aic_bic(obs, pred, sigma, k)`** computes AIC and BIC from the full Gaussian log-likelihood including the constant term $-\frac{1}{2}\sum \ln(2\pi\sigma^2)$. This is more conservative than the approximation AIC ≈ n·ln(RSS/n) + 2k and is appropriate when measurement uncertainties (σ) differ between observations.

---

### §4 — Weighted Joint Fitting and Model Comparison

**Purpose:** Estimate (α, A, U₀) by weighted least squares; assess out-of-sample performance by LOO-CV; formally compare two-parameter vs four-parameter models.

**Fitting procedure:**

`scipy.optimize.curve_fit` is used with `sigma=FU_sigma, absolute_sigma=True`. The `absolute_sigma=True` flag means the provided σ values are treated as true standard deviations (not relative weights), so the covariance matrix `pcov` yields correct standard errors in the original units. Parameter bounds `BOUNDS_2P = ([1e-3, 1e-4, 0.3], [3.0, 1e3, 3.0])` prevent non-physical solutions (α ≤ 0, A ≤ 0, U₀ < 0.3).

**LOO cross-validation:**

For each held-out condition `lo` (0 to 9), a model is refitted on the remaining 9 conditions and used to predict F and U at the held-out E value. LOO-R² is computed as:

$$\text{LOO-}R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i^{\text{LOO}})^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

LOO-R² < in-sample R² indicates overfitting. The magnitude of the difference quantifies how much predictive accuracy is lost when generalising to new conditions.

**Four-parameter model (F_4p):**

The full model is:

$$F(E) = \frac{1}{1 + \Lambda \cdot \frac{(E/E_0)^\alpha}{1 + \beta(E/E_0)^\alpha}}$$

When β → 0, this reduces to the two-parameter form. The additional parameter β represents receptor escape via dissociation or non-endocytic routing before internalisation. Its physical presence is expected (all receptor–ligand systems have non-zero off-rates), but its effect becomes detectable only when the stiffness range extends to a high-E plateau where F(E) stops decreasing.

**AIC/BIC comparison** quantifies whether the improvement in fit from adding β and a free E₀ parameter (4p model: k=5) is worth the penalty for increased complexity (2p model: k=3). ΔAIC > 0 and ΔBIC > 0 both favour the simpler model. β converging to near-zero is direct numerical evidence for non-identifiability.

**Output:** Fitted parameters with standard errors, R², LOO-R², AIC, BIC; model comparison table.

---

### §5 — Residual Diagnostics

**Purpose:** Verify that model residuals are approximately random (no systematic pattern), not serially correlated, and approximately normally distributed — assumptions that underlie the validity of the weighted least squares fit and the bootstrap CI.

**Standardised residuals** are computed as raw residuals divided by within-condition SD:

$$z_i = \frac{F_i^{\text{obs}} - F_i^{\text{pred}}}{\sigma_{F,i}}$$

Values |z| > 2 indicate that the residual exceeds two standard deviations of the measurement noise. Systematic |z| > 2 at low-E conditions (CL1/CL2) indicates structural model misfit at the soft end of the stiffness range — the power-law k_int(E) ∝ E^α approaches zero as E → 0, causing the model to under-predict F. This is a known limitation acknowledged in the parameter report (U₀ note).

**Durbin–Watson statistic:**

$$DW = \frac{\sum_{i=2}^n (e_i - e_{i-1})^2}{\sum_{i=1}^n e_i^2}$$

DW ≈ 2 indicates no serial correlation; DW < 1.5 suggests positive serial correlation (consecutive residuals have the same sign), which would indicate systematic model misspecification. DW > 1.5 for both F and U confirms the absence of systematic bias.

**Shapiro–Wilk test:** Tests H₀: residuals are normally distributed. P > 0.05 is consistent with normality. Normality of residuals is required for the standard errors from `pcov` to be valid.

**Output:** Standardised residual table; diagnostic bar chart; DW and Shapiro–Wilk results.

---

### §6 — Residual Bootstrap and Cross-Material Validation

**Purpose:** Compute bootstrap 95% CIs for (α, A, E₅₀); validate the shared-α hypothesis across NIPAM and β-glucan platforms.

**Residual bootstrap procedure:**

For each of B = 5,000 iterations:
1. Draw random indices idx_F and idx_U (independently, length 10, with replacement) from {0, …, 9}.
2. Construct bootstrap F and U vectors by adding resampled residuals to fitted values: `F_boot = F_pred + resid_F[idx_F]`.
3. Refit the joint model to the bootstrap data using the same bounds and initial parameters.
4. Store (α, A, E₅₀) for the converged fits.

**Why residual bootstrap over case bootstrap?**

In case (nonparametric) bootstrap at n = 10, each resample draws 10 indices with replacement from the original 10 conditions. The expected number of unique conditions per resample is n(1 − (1 − 1/n)^n) ≈ 6.32. This means ≈36% of the stiffness range is unrepresented in each fit, causing poorly conditioned optimisation and inflated parameter variance. Residual bootstrap preserves all E positions and only perturbs the response values, providing more stable parameter estimation.

**E₅₀ CI width:** The wide 95% CI for E₅₀ (approximately 10–56 MPa for a point estimate of ≈23 MPa) reflects mathematical amplification: E₅₀ = E₀·(1/A)^(1/α), with 1/α ≈ 6.5. A 6% bootstrap SD in A translates to a 39% SD in E₅₀ (6% × 6.5 exponent). The engineering conclusion (HβGNPs exceeds E₅₀ by >3×) remains robust despite this uncertainty.

**Cross-material validation:**

`F_joint_sys(E_all, alpha, A_N, A_B)` fits a shared α but material-specific A parameters to the pooled n = 15 dataset. The shared-α hypothesis is that the stiffness-response exponent is a property of the physical coupling mechanism (contact mechanics → internalisation rate), while A depends on surface chemistry (ligand density, glycocalyx penetration, receptor binding affinity). Per-platform independent fits provide the α values for each material alone, and their fractional difference quantifies how well the shared-α approximation holds.

**Output:** Bootstrap parameter distributions and 95% CIs; cross-material validation summary with per-platform and joint α values.

---

### §7 — Publication Figures

**Figure 1 — Model fits with bootstrap CI.** Two panels (F(E) and U(E)) show the fitted curves with bootstrap 95% CI bands, experimental data points with measurement uncertainties in both E (±SD, 60 replicates) and F/U (±SD, 3 replicates), and LOO predictions (×-markers). The E₅₀ reference is shown as a vertical dotted line.

**Figure 2 — Cross-material validation.** Experimental data from both platforms overlaid with shared-α model curves, confirming that the mechanistic framework describes both chemically distinct systems.

**Figure 3 — E–A design map.** Log-log pseudocolour heatmap of F(E, A) at the baseline α, with contour lines at F = 0.3, 0.4, 0.5, 0.6, 0.7, 0.8. The white dashed E₅₀ line and green dotted HβGNPs reference line show where the experimental material sits relative to the design threshold. Experimental data points are overlaid at the fitted A value.

All figures display inline only; no file output.

---

### §8 — Sensitivity Analysis

**Purpose:** Quantify E₅₀ robustness under ±20% perturbation of α and A. Provides the numerical basis for the claim that the HβGNPs design conclusion is robust despite uncertainty in fitted parameters.

**Joint perturbation matrix:** E₅₀ is computed for all 5×5 = 25 combinations of ±0%, ±10%, ±20% perturbation to α and A simultaneously:

```python
E50_matrix[i, j] = E50_calc(alpha_hat * (1 + pa), A_hat * (1 + pA))
```

The worst-case E₅₀ (maximum over all perturbation combinations) occurs when both α and A are decreased (α −20%, A −20%), because:
- Decreasing α (shallower stiffness response) increases E₅₀ (need higher E to achieve the same FcγR internalisation).
- Decreasing A (weaker occupancy) also increases E₅₀.

The safety margin E_HbGNPs / E₅₀_worst confirms that HβGNPs at 191.84 MPa exceeds the worst-case threshold by a factor of ≥3.4.

**Tornado chart:** Visualises the sensitivity of E₅₀ to individual ±20% perturbations of α and A. The wider bar indicates the more influential parameter.

**Output:** Joint perturbation table (5×5 absolute E₅₀ values and % change); safety margin calculation; tornado chart.

---

### §9 — Consolidated Parameter Report

**Purpose:** Complete machine-readable parameter summary for direct transcription to manuscript Supplementary Table.

Report structure:
1. Model equation and fixed E₀_ref
2. Fitted parameters (α, A, U₀) with SE and bootstrap 95% CI
3. E₅₀ with bootstrap CI and note on CI width
4. Model performance (R², LOO-R², AIC, BIC)
5. Model comparison (ΔAIC, ΔBIC, β estimate)
6. Statistical validation (ANOVA, Kruskal–Wallis, DW, Shapiro–Wilk P-values)
7. Cross-material validation (per-platform and joint α)
8. Sensitivity analysis (single and joint perturbation E₅₀ values, safety margin)
9. Note on U₀ > 1 physical interpretation

---

## Key Results Summary

| Parameter | Estimate | SE | Bootstrap 95% CI |
|-----------|----------|----|-----------------|
| α | 0.1533 | 0.0031 | — |
| A | 1.0957 | 0.0098 | — |
| U₀ | 1.2718 | 0.0064 | — |
| E₀_ref | geometric mean (fixed) | — | — |
| **E₅₀** | **22.6900 MPa** | — | **[10.9, 56.4] MPa** |

| Metric | F(E) | U(E) |
|--------|------|------|
| In-sample R² | 0.7530 | 0.8050 |
| LOO-R² | 0.6630 | 0.6790 |
| AIC (k=3) | reported | — |
| BIC (k=3) | reported | — |

| Model comparison | Value |
|-----------------|-------|
| ΔAIC (4p − 2p) | +4.0000 |
| ΔBIC (4p − 2p) | +5.9900 |
| β (4p fit) | → 0 (not identifiable) |

*Note: all values shown to 4 decimal places. Precise values are output by §9.*

---

## Engineering Design Guidance

The E–A design map (Figure 3) provides a practical tool for nanoparticle design:

1. **Determine target F value** (desired FcγR surface suppression level, e.g., F = 0.4 for 60% suppression).
2. **Estimate A** for the chosen material from a pilot experiment (3–5 stiffness conditions sufficient).
3. **Read required E** from the corresponding F = 0.4 contour on the design map.
4. **Apply safety factor** of ≥3× above the contour E value to account for parameter uncertainty (CI width).

For β-glucan NPs with A ≈ 1.10, a target F ≤ 0.5 requires E ≥ E₅₀ = 22.69 MPa. HβGNPs at 191.84 MPa provides an 8.5× margin above this threshold, consistent with the near-complete FcγR internalisation observed experimentally.
