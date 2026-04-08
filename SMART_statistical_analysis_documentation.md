# SMART Statistical Analysis — Notebook Documentation

**File:** `SMART_statistical_analysis_final.ipynb`  
**Manuscript:** Qiu et al., *Stiffness-Mediated Antibody Redistribution and Trafficking (SMART) by β-Glucan Nanomaterials Potentiates Hepatocellular Carcinoma Immunotherapy*

---

## Experimental Design

### Scientific Question

Does membrane FcγR surface redistribution (*F*) act as the dominant intermediate variable through which nanoparticle stiffness (*E*) controls αPD-L1 antibody uptake (*U*) by Kupffer cells? This notebook provides the complete statistical evidence chain to answer this question.

### Variables

| Symbol | Variable | Measurement method | Replicates |
|--------|----------|--------------------|------------|
| *E* | Young's modulus (nanoparticle stiffness) | AFM force–indentation, Hertz model fit | 60 per condition |
| *F* | Membrane FcγR surface residual (rel. Blank) | Flow cytometry, APC-anti-FcγR antibody | 3 biological replicates |
| *U* | αPD-L1 uptake by Kupffer cells (rel. Blank) | Flow cytometry, Cy5.5-αPD-L1 | 3 biological replicates |

All *F* and *U* values are normalised to the vehicle control (Blank = 1.0). Values < 1 indicate suppression relative to Blank.

### Dataset Structure

Two chemically distinct nanoparticle platforms are pooled into a single dataset of n = 15 stiffness conditions:

- **NIPAM nanoparticles:** 5 crosslinking conditions, E = 1.49–22.38 MPa. Crosslinking agent: N,N′-bis(acryloyl)cystamine (BAC); Young's modulus tuned by varying BAC concentration.
- **β-glucan nanoparticles (βGluNPs):** 10 crosslinking conditions, E = 4.10–191.84 MPa. Crosslinking agent: DTPA/EDC; Young's modulus tuned by varying DTPA molar input.

Pooling both platforms allows the analysis to span nearly two orders of magnitude in stiffness (1.49–191.84 MPa), providing sufficient range for regression-based inference. log₁₀(*E*) is used as the exposure variable throughout all regression analyses.

### Statistical Hypotheses

- **H₀ (mediation):** Nanoparticle stiffness has no indirect effect on antibody uptake through FcγR redistribution (*ab* = 0).
- **H₁ (mediation):** Stiffness influences uptake primarily via FcγR-mediated receptor trafficking (*ab* ≠ 0, proportion mediated > 50%).

---

## Software and Reproducibility

| Item | Specification |
|------|--------------|
| Language | Python 3.10 |
| Libraries | NumPy 1.24, SciPy 1.11, Pandas 2.0, Matplotlib 3.7 |
| Random seed | 2025 (all bootstrap procedures) |
| Bootstrap resamples | B = 20,000 |
| Resampling strategy | Paired percentile bootstrap (single index applied to both variables) |

---

## Cell-by-Cell Annotation

### §0 — Environment Setup

**Purpose:** Import libraries, set global constants, define formatting helper functions, configure Matplotlib.

**Key decisions:**

- `SEED = 2025` and `B_ALL = 20_000` are set as module-level constants to ensure all bootstrap procedures use identical settings.
- The colour palette (`C_NIPAM`, `C_BGLU`, `C_FIT`, `C_ACCENT`) follows Nature journal conventions and is colourblind-accessible (red/teal contrast, no pure green/red pairing).
- `fmt_p(p)` applies uniform P-value formatting: scientific notation for P < 0.0001, 4 decimal places otherwise. This is used consistently across all output in both notebooks.
- `ci4(lo, hi)` and `ci3(lo, hi)` format confidence intervals to 4 and 3 decimal places respectively — 4 d.p. for path coefficients and indirect effects, 3 d.p. for Spearman ρ and partial r.

---

### §1 — Data Import and Descriptive Statistics

**Purpose:** Read raw experimental data from Excel, aggregate replicate measurements, construct the pooled n = 15 dataset.

**Implementation details:**

`read_triplicates(sheet, cond_col, val_col)` parses replicate measurements from sheets where condition indices appear in one column and values in an adjacent column. It filters non-numeric rows and casts to correct dtypes before groupby aggregation.

`cond_stats(df)` computes group means and standard deviations (ddof=1, unbiased) by condition index.

For NIPAM NPs, raw AFM data are stored with the highest crosslinking condition in the leftmost column (order 5 → 1). The line `nipam_E_raw = nipam_E_raw[:, np.argsort(nipam_hdr)]` re-sorts to ascending stiffness order before computing means. Failure to apply this re-sort would invert the stiffness-response relationship for NIPAM data.

**Output:** Summary table with E mean ± SEM (60 replicates), F mean ± SD (3 replicates), U mean ± SD (3 replicates) for all 15 conditions.

---

### §2 — Spearman Rank Correlation

**Purpose:** Quantify pairwise monotonic associations among *E*, *F*, and *U* without assuming linearity. This establishes that ρ(*F*, *U*) > ρ(*E*, *U*), motivating the subsequent mediation framework.

**Mathematical basis:**

Spearman's ρ is computed as the Pearson correlation of rank-transformed variables. For paired observations (*x*ᵢ, *y*ᵢ), this is equivalent to:

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

where $d_i = \text{rank}(x_i) - \text{rank}(y_i)$.

**Bootstrap implementation:**

`spearman_boot(x, y, B, seed)` draws B paired bootstrap samples using a single index vector applied simultaneously to both x and y:

```python
x[idx := rng.choice(nn, nn, replace=True)], y[idx]
```

This **paired resampling** is critical: it preserves the bivariate structure of the data. The alternative (independent resampling of x and y with different index vectors) would artificially inflate the bootstrap variance by treating the pair as two independent univariate samples. The resulting CI would be too wide, understating the precision of the correlation estimate.

**Output:** Spearman ρ, bootstrap SD, 95% CI, and P-value for five pairs: E vs U, E vs F, F vs U, log₁₀E vs U, log₁₀E vs F.

---

### §3 — Partial Correlation and Variance Decomposition

**Purpose:** Quantify the independent contribution of *F* to predicting *U* after removing the shared variance with *E* (and vice versa). This addresses the confound that E and F are themselves correlated (ρ ≈ −0.83), which inflates simple bivariate associations.

**Three complementary methods are implemented:**

**Method 1 — Partial correlation.** Both the predictor of interest and *U* are residualised on the covariate by OLS before computing Pearson r:

```
r(F→U | logE) = corr(resid(F ~ logE),  resid(U ~ logE))
r(E→U | F)   = corr(resid(logE ~ F),  resid(U ~ F))
```

`ols_resid(y, *covariates)` computes OLS residuals by solving the normal equations via `numpy.linalg.lstsq`. The intercept column (column of ones) is always included.

**Method 2 — Semi-partial R² (variance decomposition).** The unique contribution of each predictor is the increment in R² when that predictor is added to a model already containing the other:

$$\Delta R^2(F) = R^2(E+F) - R^2(E\text{ alone}), \quad \Delta R^2(E) = R^2(E+F) - R^2(F\text{ alone})$$

Shared variance = R²(E) + R²(F) − R²(E+F), arising from predictor collinearity.

**Method 3 — Standardised regression coefficients (β).** All variables are Z-score normalised before OLS regression. The resulting coefficients β_E and β_F are dimensionless and directly comparable as effect sizes.

**Bootstrap:** All three metrics use B = 20,000 paired resamples. The bootstrap loop recomputes all metrics simultaneously from each resampled dataset, then extracts percentile CIs from the stored distributions.

**Output:** Partial r values, unique R² contributions with percentages, standardised β coefficients — all with 95% CIs.

---

### §4 — Bootstrap Mediation Analysis

**Purpose:** Formally test whether FcγR redistribution (*F*) is a statistically significant mediator of the stiffness (*E*) → uptake (*U*) relationship, and estimate the proportion of the total effect transmitted through this pathway.

**Path model (Baron–Kenny OLS framework):**

`fit_mediation(lE, Fv, Uv)` estimates three OLS regression equations:

| Equation | Regression | Coefficient extracted |
|----------|-----------|----------------------|
| (i)  | F ~ intercept + log₁₀E | path *a* (index [1]) |
| (ii) | U ~ intercept + log₁₀E + F | path *b* (index [2]); direct effect *c'* (index [1]) |
| (iii)| U ~ intercept + log₁₀E | total effect *c* (index [1]) |

The indirect effect is *ab* = *a* × *b*. Its sign must be checked: *a* < 0 (stiffer → lower FcγR surface residual) and *b* > 0 (higher FcγR residual → more antibody uptake), so *ab* < 0, indicating that increased stiffness suppresses uptake via FcγR internalization — consistent with the biological mechanism.

**Bootstrap significance test:**

Mediation is confirmed by the percentile bootstrap 95% CI of *ab*. If the CI entirely excludes zero, the indirect effect is statistically significant at α = 0.05. The two-sided approximate bootstrap P-value is computed as:

$$P_{ab} = 2 \min\left(\Pr(ab^* \leq 0),\; \Pr(ab^* \geq 0)\right)$$

where $ab^*$ is the bootstrap distribution of the indirect effect.

**Proportion mediated:** *ab* / *c*, where *c* is the total effect. Values > 0.5 indicate that FcγR redistribution accounts for the majority of the stiffness–uptake relationship.

**Significance stars:** `sig_star(ci_lo, ci_hi, p)` returns `***` (P < 0.001), `**` (P < 0.01), `*` (P < 0.05), or `ns` based on whether the CI excludes zero and the P-value threshold.

**Output:** Point estimates and 95% CIs for paths a, b, c', c, indirect effect ab, and proportion mediated.

---

### §5 — Publication Figures

Three figures are generated for manuscript submission.

**Figure 1 — Pseudocolour heatmap.** A 300×300 grid is interpolated from the 15 (E, F) data points using `scipy.interpolate.griddata` with cubic method. The heatmap shows *U* as a function of both *E* (x-axis) and *F* (y-axis), visualising the joint stiffness–receptor–uptake relationship. Contour lines are overlaid and labelled. Data points are coloured by platform (NIPAM = red, βGlu = teal).

**Figure 2 — Partial regression (added-variable) plots.** Two panels show the net relationship of each predictor with *U* after mutual adjustment. These plots provide visual confirmation that both F→U (panel b) and E→U (panel a) retain independent associations, but the slope and scatter of panel (b) are markedly tighter, consistent with F being the dominant predictor.

**Figure 3 — Bootstrap distribution and path diagram.** Panel (a) shows the bootstrap histogram of *ab* with the point estimate and 95% CI indicated. Panel (b) shows the OLS path diagram with nodes (X = log₁₀E, F = FcγR, Y = uptake) connected by arrows labelled with path coefficients and 95% CIs. The indirect effect summary box is placed at the top.

All figures display inline only; no file output.

---

### §6 — Consolidated Parameter Report

**Purpose:** Machine-readable summary of all estimated parameters with 4 decimal place precision, suitable for direct transcription to the manuscript Supplementary Table.

The report is structured into five blocks: Spearman correlations, partial correlations, variance decomposition, standardised coefficients, and mediation analysis. All P-values use the unified `fmt_p()` formatting. The final interpretation block summarises the key conclusions in quantitative terms.

---

## Statistical Methods (Manuscript Text)

> All statistical analyses were performed in Python 3.10 using NumPy (v1.24), SciPy (v1.11), Pandas (v2.0), and Matplotlib (v3.7). The pooled dataset comprised *n* = 15 crosslinking conditions across two nanoparticle platforms (NIPAM NPs: 5 conditions; βGluNPs: 10 conditions). Young's modulus (*E*) was taken as the mean of 60 AFM force–indentation measurements per condition; αPD-L1 uptake (*U*) and membrane FcγR residual (*F*) were each taken as the mean of three independent biological replicates per condition. All regression analyses used log₁₀-transformed *E* (log₁₀E) to accommodate the multi-order-of-magnitude range of stiffness values.
>
> Pairwise monotonic associations were quantified by Spearman's ρ with bootstrap 95% CIs (B = 20,000; seed = 2025; paired percentile resampling). Independent contributions of *F* and *E* to *U* were assessed by: (i) partial correlations (OLS residualisation); (ii) semi-partial R² variance decomposition; and (iii) standardised regression coefficients. Mediation analysis used the Baron–Kenny OLS path-model framework; indirect effect *ab* significance was assessed by non-parametric bootstrap (B = 20,000). All analyses are fully reproducible from the archived notebook.

---

## Key Results Summary

| Analysis | Estimate | 95% CI | P-value |
|----------|----------|--------|---------|
| ρ(F, U) | +0.9430 | [+0.764, +0.996] | < 0.0001 |
| ρ(E, U) | −0.8540 | [−0.989, −0.515] | < 0.0001 |
| Partial r(F→U \| log₁₀E) | +0.9010 | [+0.797, +0.964] | 0.0000 |
| Partial r(E→U \| F) | −0.7530 | [−0.913, −0.448] | 0.0012 |
| Unique R²: F | 0.2520 (26.7%) | [0.089, 0.508] | — |
| Unique R²: E | 0.0770 (8.1%) | [0.010, 0.206] | — |
| β_F | +0.6780 | [+0.502, +0.864] | — |
| β_E | −0.3740 | [−0.563, −0.156] | — |
| Path a | −0.2560 | [−0.389, −0.150] | — |
| Path b | +0.4300 | [+0.315, +0.581] | — |
| Indirect effect ab | −0.1100 | [−0.185, −0.050] | 0.0030 |
| Total effect c | −0.2010 | [−0.270, −0.142] | — |
| Direct effect c' | −0.0900 | [−0.138, −0.040] | — |
| Proportion mediated | 54.97% | [33.53%, 80.46%] | — |

*Note: all values shown to 4 significant figures. Precise 4-decimal-place values are output by the notebook's consolidated report cell.*
