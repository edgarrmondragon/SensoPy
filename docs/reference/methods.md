# Methods

Every method is a subclass of `sensopy.discrimination.DiscriminationMethod` and exposes
the same interface. Ready-made instances are provided as module-level constants in
`sensopy.discrimination`.

| Constant             | Class                   | Guessing probability $P_g$ | $P_c$ at d′ = 1 |
| -------------------- | ----------------------- | -------------------------- | --------------- |
| `TWO_AFC`            | `TwoAFCMethod`          | 1/2                        | 0.760           |
| `DUAL_PAIR`          | `DualPairMethod`        | 1/2                        | 0.573           |
| `DUO_TRIO`           | `DuoTrioMethod`         | 1/2                        | 0.583           |
| `THREE_AFC`          | `ThreeAFCMethod`        | 1/3                        | 0.634           |
| `FOUR_AFC`           | `FourAFCMethod`         | 1/4                        | 0.552           |
| `TRIANGLE`           | `TriangleMethod`        | 1/3                        | 0.418           |
| `SPECIFIED_TETRAD`   | `SpecifiedTetradMethod` | 1/6                        | 0.463           |
| `UNSPECIFIED_TETRAD` | `UnspecifiedTetrad`     | 1/3                        | 0.494           |

Two parameterised methods are available as classes: `MultipleAFCMethod(m)` for any
m-AFC, and `MPlusNMethod(m, n, specified=False, seed=None)`, whose psychometric
function is estimated by Monte Carlo simulation (pass `seed` for reproducibility).

## `DiscriminationMethod`

`psychometric_function(d)`
:   Probability $P_c$ of a correct response when the true distance is `d`.

`guessing`
:   Property. Probability $P_g$ of a correct response by chance (`d = 0`).

`discriminators(d)`
:   Proportion of discriminators, $p_d = (P_c - P_g) / (1 - P_g)$.

`simulate(d, n, size=..., rng=...)`
:   Draw simulated counts of correct responses for `n` panelists.

## `DiscriminationTest`

`DiscriminationTest(method)`
:   Create a test for one method.

`difference(x, n, pd0=0, conf_level=0.95)`
:   One-tailed test of $H_0: p_d \le p_{d0}$ against $H_1: p_d > p_{d0}$. Returns
    [`TestResults`](results.md).

`equivalence(x, n, pd0=0, conf_level=0.95)`
:   One-tailed test of $H_0: p_d \ge p_{d0}$ against $H_1: p_d < p_{d0}$. Returns
    [`TestResults`](results.md).

`simulate(d, n, size=1000, *, kind="difference", pd0=0, conf_level=0.95, rng=None)`
:   Simulate `size` tests with true distance `d` and return
    [`SimulationResults`](results.md#simulationresults). `kind` is `"difference"` or
    `"equivalence"`; a `ValueError` is raised for an unknown `kind` or `size < 1`.

The parameters are: `x`, number of correct responses; `n`, number of panelists;
`pd0`, the null-hypothesis proportion of discriminators; `conf_level`, the confidence
level of the interval estimates (significance level $\alpha = 1 - $ `conf_level`).
