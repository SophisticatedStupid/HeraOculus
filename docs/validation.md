# Trend / Extreme-Value Cross-Validation

Generated: 2026-11-23T13:01:53.287201+00:00Z

## Scope

In-house implementations are cross-checked against independent reference
implementations on seeded synthetic series and (when network is available) a
real public dataset (NOAA Mauna Loa monthly CO2).

## Documented tolerances

Tolerances are fixed and are never loosened to hide a mismatch. Any row
with status `FAIL` is a genuine discrepancy requiring investigation.

- **mk_z**: 1e-06
- **mk_p**: 1e-06
- **mk_s**: 0.0
- **mk_var_s**: 1e-09
- **sen_slope**: 1e-09
- **hrao_p**: 0.02
- **hrao_z**: 0.05
- **hrao_var_s_ratio**: 0.1
- **gev_xi**: 1e-06
- **gev_mu**: 1e-06
- **gev_sigma**: 1e-06
- **gev_return_level**: 0.0001

## Reference implementations

- Mann-Kendall / Theil-Sen / Hamed-Rao: [`pymannkendall`](https://pypi.org/project/pymannkendall/) (original_test, sens_slope, hamed_rao_modification_test)
- GEV: [`scipy.stats.genextreme.fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html)

## Results

| Method | Series | Statistic | Ours | Reference | Abs diff | Tolerance | Status |
|--------|--------|-----------|------|-----------|----------|-----------|--------|
| Mann-Kendall | synthetic_increasing | Z | 7.98095867 | 7.98095867 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_increasing | p | 0.0 | 0.0 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_increasing | S | 686.0 | 686.0 | 0.0 | 0.0 | PASS |
| Mann-Kendall | synthetic_increasing | var_S | 7366.66666667 | 7366.66666667 | 0.0 | 1e-09 | PASS |
| Theil-Sen | synthetic_increasing | slope | 0.26239787 | 0.26239787 | 0.0 | 1e-09 | PASS |
| Hamed-Rao MK | synthetic_increasing | Z | 7.98095867 | 7.98095867 | 0.0 | 0.05 | PASS |
| Hamed-Rao MK | synthetic_increasing | p | 0.0 | 0.0 | 0.0 | 0.02 | PASS |
| Hamed-Rao MK | synthetic_increasing | var_S_ratio | 1.0 | 1.0 | 0.0 | 0.1 | PASS |
| Mann-Kendall | synthetic_decreasing | Z | -7.07217798 | -7.07217798 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_decreasing | p | 0.0 | 0.0 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_decreasing | S | -608.0 | -608.0 | 0.0 | 0.0 | PASS |
| Mann-Kendall | synthetic_decreasing | var_S | 7366.66666667 | 7366.66666667 | 0.0 | 1e-09 | PASS |
| Theil-Sen | synthetic_decreasing | slope | -0.18215588 | -0.18215588 | 0.0 | 1e-09 | PASS |
| Hamed-Rao MK | synthetic_decreasing | Z | -7.07217798 | -7.07217798 | 0.0 | 0.05 | PASS |
| Hamed-Rao MK | synthetic_decreasing | p | 0.0 | 0.0 | 0.0 | 0.02 | PASS |
| Hamed-Rao MK | synthetic_decreasing | var_S_ratio | 1.0 | 1.0 | 0.0 | 0.1 | PASS |
| Mann-Kendall | synthetic_none | Z | 0.01165103 | 0.01165103 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_none | p | 0.99070403 | 0.99070403 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_none | S | 2.0 | 2.0 | 0.0 | 0.0 | PASS |
| Mann-Kendall | synthetic_none | var_S | 7366.66666667 | 7366.66666667 | 0.0 | 1e-09 | PASS |
| Theil-Sen | synthetic_none | slope | 0.00017866 | 0.00017866 | 0.0 | 1e-09 | PASS |
| Hamed-Rao MK | synthetic_none | Z | 0.01143148 | 0.01143148 | 0.0 | 0.05 | PASS |
| Hamed-Rao MK | synthetic_none | p | 0.9908792 | 0.9908792 | 0.0 | 0.02 | PASS |
| Hamed-Rao MK | synthetic_none | var_S_ratio | 1.03878189 | 1.03878189 | 0.0 | 0.1 | PASS |
| Mann-Kendall | synthetic_tied | Z | 8.50009878 | 8.50009878 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_tied | p | 0.0 | 0.0 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_tied | S | 727.0 | 727.0 | 0.0 | 0.0 | PASS |
| Mann-Kendall | synthetic_tied | var_S | 7295.0 | 7295.0 | 0.0 | 1e-09 | PASS |
| Theil-Sen | synthetic_tied | slope | 0.3125 | 0.3125 | 0.0 | 1e-09 | PASS |
| Hamed-Rao MK | synthetic_tied | Z | 8.50009878 | 8.50009878 | 0.0 | 0.05 | PASS |
| Hamed-Rao MK | synthetic_tied | p | 0.0 | 0.0 | 0.0 | 0.02 | PASS |
| Hamed-Rao MK | synthetic_tied | var_S_ratio | 1.0 | 1.0 | 0.0 | 0.1 | PASS |
| Mann-Kendall | synthetic_ar1_trend | Z | 5.6973559 | 5.6973559 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_ar1_trend | p | 1e-08 | 1e-08 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_ar1_trend | S | 490.0 | 490.0 | 0.0 | 0.0 | PASS |
| Mann-Kendall | synthetic_ar1_trend | var_S | 7366.66666667 | 7366.66666667 | 0.0 | 1e-09 | PASS |
| Theil-Sen | synthetic_ar1_trend | slope | 0.16433292 | 0.16433292 | 0.0 | 1e-09 | PASS |
| Hamed-Rao MK | synthetic_ar1_trend | Z | 3.51233285 | 3.51233285 | 0.0 | 0.05 | PASS |
| Hamed-Rao MK | synthetic_ar1_trend | p | 0.00044419 | 0.00044419 | 0.0 | 0.02 | PASS |
| Hamed-Rao MK | synthetic_ar1_trend | var_S_ratio | 2.63120913 | 2.63120913 | 0.0 | 0.1 | PASS |
| Mann-Kendall | synthetic_short | Z | 0.0 | 0.0 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_short | p | 1.0 | 1.0 | 0.0 | 1e-06 | PASS |
| Mann-Kendall | synthetic_short | S | 1.0 | 1.0 | 0.0 | 0.0 | PASS |
| Mann-Kendall | synthetic_short | var_S | 3.66666667 | 3.66666667 | 0.0 | 1e-09 | PASS |
| Theil-Sen | synthetic_short | slope | 0.25 | 0.25 | 0.0 | 1e-09 | PASS |
| GEV | synthetic_gev_block_1 | shape_xi | -0.14206116 | -0.14206116 | 0.0 | 1e-06 | PASS |
| GEV | synthetic_gev_block_1 | location_mu | 9.78839729 | 9.78839729 | 0.0 | 1e-06 | PASS |
| GEV | synthetic_gev_block_1 | scale_sigma | 1.97724331 | 1.97724331 | 0.0 | 1e-06 | PASS |
| GEV | synthetic_gev_block_1 | return_level_T50 | 15.71108414 | 15.71108414 | 0.0 | 0.0001 | PASS |
| GEV | synthetic_gev_block_2 | shape_xi | 0.41120535 | 0.41120535 | 0.0 | 1e-06 | PASS |
| GEV | synthetic_gev_block_2 | location_mu | 5.06830357 | 5.06830357 | 0.0 | 1e-06 | PASS |
| GEV | synthetic_gev_block_2 | scale_sigma | 1.50020492 | 1.50020492 | 0.0 | 1e-06 | PASS |
| GEV | synthetic_gev_block_2 | return_level_T50 | 19.57165806 | 19.57165806 | 0.0 | 0.0001 | PASS |
| GEV | synthetic_gev_block_3 | shape_xi | -0.03381836 | -0.03381836 | 0.0 | 1e-06 | PASS |
| GEV | synthetic_gev_block_3 | location_mu | 0.00859275 | 0.00859275 | 0.0 | 1e-06 | PASS |
| GEV | synthetic_gev_block_3 | scale_sigma | 1.08427138 | 1.08427138 | 0.0 | 1e-06 | PASS |
| GEV | synthetic_gev_block_3 | return_level_T50 | 3.9720972 | 3.9720972 | 0.0 | 0.0001 | PASS |
| GEV | real_mauna_loa_co2_annual_max | shape_xi | -0.04029931 | -0.04029931 | 0.0 | 1e-06 | PASS |
| GEV | real_mauna_loa_co2_annual_max | location_mu | 346.85889886 | 346.85889886 | 0.0 | 1e-06 | PASS |
| GEV | real_mauna_loa_co2_annual_max | scale_sigma | 28.16519184 | 28.16519184 | 0.0 | 1e-06 | PASS |
| GEV | real_mauna_loa_co2_annual_max | return_level_T50 | 448.55284129 | 448.55284129 | 0.0 | 0.0001 | PASS |

## Summary

- Comparisons run: 61
- Passed: 61
- Failed (discrepancy): 0
- Skipped: 0

> **Result: PASS** — all comparisons within documented tolerance.
