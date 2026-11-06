# Cite SensoPy

If SensoPy contributed to results you publish, please cite it.

## Get a ready-made citation

The repository ships a [`CITATION.cff`](https://github.com/edgarrmondragon/SensoPy/blob/main/CITATION.cff)
file in the [Citation File Format](https://citation-file-format.github.io/). GitHub reads
it and adds a **Cite this repository** button to the repository page, which offers
APA and BibTeX versions of the citation.

## Cite from the command line

[`cffconvert`](https://github.com/citation-file-format/cffconvert) converts the file to
other formats:

```shell
uvx cffconvert --format bibtex
uvx cffconvert --format apalike
```

Run these from a checkout of the repository.

## Pin the version

Add the version you used so that others can reproduce your analysis:

```python
import importlib.metadata

print(importlib.metadata.version("sensopy"))
```

## Cite the underlying methods

SensoPy implements methods from Bi (2015). Cite the book as well when you report
results computed with the library:

> Bi, J. (2015). *Sensory Discrimination Tests and Measurements: Sensometrics in
> Sensory Evaluation* (2nd ed.). Wiley-Blackwell.
