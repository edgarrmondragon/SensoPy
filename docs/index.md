# SensoPy

SensoPy is a Python library for the design and analysis of sensory discrimination
tests: triangle, duo-trio, 2-AFC, tetrad and friends. Given a count of correct
responses, it estimates how different two products are and tests whether that
difference is real, or whether the products are similar enough to be interchangeable.

The documentation is organised by what you are trying to do.

<div class="grid cards" markdown>

- **Tutorials**

    ______________________________________________________________________

    Learn by doing. Start here if you are new to SensoPy.

    [Run your first difference test](tutorials/first-difference-test.md)

- **How-to guides**

    ______________________________________________________________________

    Recipes for specific tasks.

    [Test for equivalence](how-to/equivalence-test.md) ·
    [Estimate power by simulation](how-to/estimate-power.md)

- **Reference**

    ______________________________________________________________________

    Precise descriptions of the methods and result objects.

    [Methods](reference/methods.md) · [Results](reference/results.md)

- **Explanation**

    ______________________________________________________________________

    Background on the statistics behind the library.

    [The Thurstonian model](explanation/thurstonian-model.md) ·
    [Hypothesis tests and power](explanation/hypothesis-tests-and-power.md)

</div>

## Installation

```shell
pip install git+https://github.com/edgarrmondragon/SensoPy.git
```

SensoPy requires Python 3.13 or newer.

## References

Methods follow Bi, J. (2015), *Sensory Discrimination Tests and Measurements:
Sensometrics in Sensory Evaluation*, 2nd ed., Wiley-Blackwell.
