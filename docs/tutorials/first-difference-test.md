# Run your first difference test

In this tutorial you will analyse the results of a triangle test and learn how to read
the output. It takes about five minutes.

## The experiment

You have reformulated a product and want to know whether consumers can tell the new
version from the old one. You run a **triangle test**: each of 30 panelists receives
three samples, two of one product and one of the other, and picks the odd one out.
19 of them pick correctly.

Just by guessing, a panelist is right a third of the time, so 10 correct answers out of
30 would be unremarkable. Is 19 enough to conclude the products differ?

## Run the test

Create a `DiscriminationTest` for the triangle method and call `difference()` with the
number of correct responses and the number of panelists:

```python
from sensopy.discrimination import TRIANGLE, DiscriminationTest

test = DiscriminationTest(TRIANGLE)
result = test.difference(19, 30)
```

## Read the results

```python
print(f"d'      = {result.d_prime.estimate:.4f}  [{result.d_prime.lower:.4f}, {result.d_prime.upper:.4f}]")
print(f"p_d     = {result.pd.estimate:.4f}  [{result.pd.lower:.4f}, {result.pd.upper:.4f}]")
print(f"p-value = {result.p_value:.4f}")
```

```
d'      = 2.1462  [1.1262, 3.1336]
p_d     = 0.4500  [0.1578, 0.7011]
p-value = 0.0007
```

- The **p-value** of 0.0007 is far below 0.05, so you reject the hypothesis that the
    products are indistinguishable.
- **d′ = 2.15** is the estimated sensory distance between the products. Unlike the raw
    proportion correct, it does not depend on which test method you used. See
    [The Thurstonian model](../explanation/thurstonian-model.md).
- **p_d = 0.45** says roughly 45% of the panel can reliably detect the difference; the
    rest were guessing.

Each estimate comes with a 95% confidence interval. The intervals are wide because 30
panelists is a small sample, which is worth keeping in mind before trusting d′ to two
decimal places.

## Try another method

The same data would mean something different under another method, because guessing
is easier or harder. Swap the method and compare:

```python
from sensopy.discrimination import TWO_AFC

DiscriminationTest(TWO_AFC).difference(19, 30).d_prime.estimate
```

A 2-AFC panelist guesses right half the time, so 19/30 correct implies a much smaller
d′ than it does for the triangle test.

## Next steps

- Check whether two products are *similar* instead: [Test for equivalence](../how-to/equivalence-test.md).
- Plan a panel size: [Estimate power by simulation](../how-to/estimate-power.md).
- Understand what the numbers mean: [Hypothesis tests and power](../explanation/hypothesis-tests-and-power.md).
