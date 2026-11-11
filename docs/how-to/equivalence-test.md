# Test for equivalence

Use an equivalence test when you want evidence that two products are *similar*, for
example after a cost-saving ingredient swap, rather than evidence that they differ.

## Choose the threshold

Equivalence is always relative to a threshold `pd0`: the largest proportion of
discriminators you are willing to tolerate and still call the products equivalent.
Pick it before collecting data. Values between 0.2 and 0.3 are common, but the right
value depends on how costly a noticeable change would be.

## Run the test

```python
from sensopy.discrimination import DiscriminationTest, TWO_AFC

test = DiscriminationTest(TWO_AFC)
result = test.equivalence(16, 30, pd0=0.30)

print(f"p-value = {result.p_value:.4f}")
```

```
p-value = 0.1263
```

The null hypothesis is that the proportion of discriminators is *at least* `pd0`.
A p-value below your significance level (`1 - conf_level`, 0.05 by default) lets you
conclude the products are equivalent. Here 0.13 is not small enough, so the data do
not demonstrate equivalence.

## Change the confidence level

```python
result = test.equivalence(16, 30, pd0=0.30, conf_level=0.90)
```

!!! warning

    Failing to reject is not the same as showing the products differ. With a small
    panel an equivalence test often lacks the power to succeed even when the products
    are truly similar. See [Estimate power by simulation](estimate-power.md).
