# Results

All result types are dataclasses defined in `sensopy.discrimination`.

## `Statistic`

A point estimate with its uncertainty.

| Field      | Description                            |
| ---------- | -------------------------------------- |
| `estimate` | Point estimate                         |
| `stderr`   | Standard error                         |
| `lower`    | Lower limit of the confidence interval |
| `upper`    | Upper limit of the confidence interval |

## `TestResults`

Returned by `DiscriminationTest.difference()` and `DiscriminationTest.equivalence()`.

| Field     | Type        | Description                                          |
| --------- | ----------- | ---------------------------------------------------- |
| `pg`      | `float`     | Guessing probability of the method                   |
| `pc`      | `Statistic` | Proportion of correct responses                      |
| `pd`      | `Statistic` | Proportion of discriminators                         |
| `d_prime` | `Statistic` | Thurstonian distance d′                              |
| `p_value` | `float`     | One-tailed exact binomial p-value                    |
| `alpha`   | `float`     | Significance level, `1 - conf_level`                 |
| `power`   | `float`     | Power of the test at the observed proportion correct |

## `SimulationResults`

Returned by `DiscriminationTest.simulate()`.

| Field      | Type             | Description                                          |
| ---------- | ---------------- | ---------------------------------------------------- |
| `x`        | `NDArray[int]`   | Simulated number of correct responses per experiment |
| `p_values` | `NDArray[float]` | p-value of each simulated experiment                 |
| `alpha`    | `float`          | Significance level                                   |
| `power`    | `float`          | Fraction of experiments with `p_value <= alpha`      |
