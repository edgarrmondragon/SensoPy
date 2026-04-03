# SensoPy

[![Tests](https://github.com/edgarrmondragon/SensoPy/actions/workflows/tests.yml/badge.svg)](https://github.com/edgarrmondragon/SensoPy/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/edgarrmondragon/SensoPy/branch/master/graph/badge.svg?token=k3m3CmIACa)](https://codecov.io/gh/edgarrmondragon/SensoPy)

Python library for the design and analysis of sensory discrimination tests. I have plans to make this a full-fledged sensory data analysis library by including all common standard techniques (Preference Mapping, Penalty Analysis, DOE, etc.)

## Installation

`pip install git+https://github.com/edgarrmondragon/SensoPy.git`

## Usage

### Difference test

One-tailed test of H0: p_d = 0 (no difference) against H1: p_d > 0 (a difference
exists). 19 correct responses out of 30 panelists using the Triangle method:

```python
from sensopy import DiscriminationTest
from sensopy.discrimination import TRIANGLE

test = DiscriminationTest(TRIANGLE)
result = test.difference(19, 30)

print(f"d'      = {result.d_prime.estimate:.4f}  [{result.d_prime.lower:.4f}, {result.d_prime.upper:.4f}]")
print(f"p_d     = {result.pd.estimate:.4f}  [{result.pd.lower:.4f}, {result.pd.upper:.4f}]")
print(f"p-value = {result.p_value:.4f}")
```

```
d'      = 2.1462  [1.1262, 3.1336]
p_d     = 0.4500  [0.1578, 0.7011]
p-value = 0.0007
```

- **d' = 2.15** (95% CI: [1.13, 3.13]): estimated Thurstonian discriminal distance,
  a method-independent measure of the sensory difference between the two products.
- **p_d = 0.45** (95% CI: [0.16, 0.70]): an estimated 45% of the panel can reliably
  detect the difference.
- **p-value = 0.0007**: strong evidence against H0; the products are perceptibly
  different at any conventional significance level.

### Equivalence test

One-tailed test of H0: p_d >= 0.30 against H1: p_d < 0.30 (the products are
sensorially similar). `pd0 = 0.30` sets the equivalence threshold: the maximum
proportion of discriminators we are willing to tolerate and still call the products
equivalent. 16 correct out of 30 panelists using the 2-AFC method:

```python
from sensopy import DiscriminationTest
from sensopy.discrimination import TWO_AFC

test = DiscriminationTest(TWO_AFC)
result = test.equivalence(16, 30, pd0=0.30)

print(f"d'      = {result.d_prime.estimate:.4f}  [{result.d_prime.lower:.4f}, {result.d_prime.upper:.4f}]")
print(f"p_d     = {result.pd.estimate:.4f}  [{result.pd.lower:.4f}, {result.pd.upper:.4f}]")
print(f"p-value = {result.p_value:.4f}")
```

```
d'      = 0.1183  [0.0000, 0.8099]
p_d     = 0.0667  [0.0000, 0.4332]
p-value = 0.1263
```

- **d' = 0.12** (95% CI: [0.00, 0.81]): small estimated sensory distance, consistent
  with near-identical products.
- **p_d = 0.07** (95% CI: [0.00, 0.43]): only ~7% of the panel estimated to reliably
  discriminate the products.
- **p-value = 0.13**: insufficient evidence to conclude equivalence at α = 0.05 with
  pd0 = 0.30 (i.e., we cannot rule out that ≥ 30% of consumers can detect the
  difference); a larger panel would be needed to meet that threshold.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Credits

<ul>
  <li>Edgar Ramírez Mondragón</li>
</ul>

## References

<ol>
  <li>Bi, J. 2015. Sensory Discrimination Tests and Measurements. <em>Sensometrics in Sensory Evaluation</em>. Wiley Blackwell, Richmond, Virginia, USA.</li>
</ol>
