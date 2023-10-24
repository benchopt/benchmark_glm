Benchmark for Generalized Linear Models (GLM)
=============================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
The GLM consists in solving the following program:

$\\min_{w} f(W^\\top X, y) + \\lambda R(w)$

where $f$ is a link function, defined by the distribution of the target variables,
and $R$ is a regularisation function -- typically the $\\ell_1$ or $\\ell_2$ norm.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_glm
   $ benchopt run benchmark_glm

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_glm -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_glm/actions/workflows/main.yml/badge.svg
   :target: https://github.com/benchopt/benchmark_glm/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
