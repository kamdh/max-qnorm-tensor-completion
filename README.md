# Max-quasinorm tensor completion in `python`
---

Tensor completion code to accompany https://arxiv.org/abs/1910.10692.

Solves the following problem:

[//]: # (\mathrm{min}_{T = U^{(1)} \circ \cdots \circ} \|T\|_\mathrm{max} \qquad \mathrm{ s.t. } \qquad \| \Omega * (T - Z) \|_F \leq \delta)
![\mathrm{min}_{T = U^{(1)} \circ \cdots \circ} \|T\|_\mathrm{max} \qquad \mathrm{ s.t. } \qquad \| \Omega * (T - Z) \|_F \leq \delta](https://latex.codecogs.com/svg.latex?%5Cmathrm%7Bmin%7D_%7BT%20%3D%20U%5E%7B(1)%7D%20%5Ccirc%20%5Ccdots%20%5Ccirc%7D%20%5C%7CT%5C%7C_%5Cmathrm%7Bmax%7D%20%5Cquad%20%5Cmathrm%7B%20s.t.%20%7D%20%5Cquad%20%5C%7C%20%5COmega%20*%20(T%20-%20Z)%20%5C%7C_F%20%5Cleq%20%5Cdelta)

via the relaxation

[//]: # (\mathrm{min}_{T = U^{(1)} \circ \cdots \circ} \|T\|_\mathrm{max} + \frac{\kappa}{2} \| \Omega * (T - Z - R) \|_F^2 + \frac{\beta}{2} \|R\|_F^2 \qquad \mathrm{ s.t. } \qquad \| R \|_F \leq \delta)
![\mathrm{min}_{T = U^{(1)} \circ \cdots \circ} \|T\|_\mathrm{max} + \frac{\kappa}{2} \| \Omega * (T - Z - R) \|_F^2 + \frac{\beta}{2} \|R\|_F^2 \qquad \mathrm{ s.t. } \qquad \| R \|_F \leq \delta](https://latex.codecogs.com/svg.latex?%5Cmathrm%7Bmin%7D_%7BT%20%3D%20U%5E%7B(1)%7D%20%5Ccirc%20%5Ccdots%20%5Ccirc%7D%20%5C%7CT%5C%7C_%5Cmathrm%7Bmax%7D%20%2B%20%5Cfrac%7B%5Ckappa%7D%7B2%7D%20%5C%7C%20%5COmega%20*%20(T%20-%20Z%20-%20R)%20%5C%7C_F%5E2%20%2B%20%5Cfrac%7B%5Cbeta%7D%7B2%7D%20%5C%7CR%5C%7C_F%5E2%20%5Cqquad%20%5Cmathrm%7B%20s.t.%20%7D%20%5Cqquad%20%5C%7C%20R%20%5C%7C_F%20%5Cleq%20%5Cdelta)

Requirements
---

* python 3.7
* numpy, scipy, sparse, itertools, numpy_groupies