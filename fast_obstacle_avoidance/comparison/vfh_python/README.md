# Vector Field Histogram - Python Implementation

Author: Zhanwen (Phil) Chen

Vector Field Histogram is a robot path planning algorithm. We originally
implemented this algorithm in C++ in [VectorFieldHistogramTesting](https://github.com/vanderbiltrobotics/VectorFieldHistogramTesting).

We reimplement VFH because the original C++ implementation contains a
mystery bug, rendering it unusable. We suspect that the bug may be a result
of wrong C++ pointers, and Python helps avoid such idiosyncracies. Another
consideration is that Phil sucks at C++ but no one sucks at Python.

"What about production?" You ask. Python can be compiled into C++ code with
`Cython`, and there are other possible hacks to jam this into ROSMOD.


# TODOs

- [x] (Fixed) BUG: Histogram grid active region is upside down. Potential cause: have been using \[x\]\[y\] but should have been \[y\]\[x\] (it was the plotting x.append(y))
- [] FIXME: Polar Histogram pie chart switch angle and maybe counterclockwise.
