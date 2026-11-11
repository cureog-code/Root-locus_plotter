# Root Locus Plotter
A Python implementation of a root-locus plotter for characteristic equations of the form

$$
D(s) + K N(s) = 0, \qquad K \geq 0.
$$

Enter the real polynomial coefficients of $D(s)$ and $N(s)$; the program computes the characteristic roots across a range of gains and plots their trajectories in the complex $s$-plane. It uses numerical root solving, adaptive gain sampling, and the Hungarian assignment algorithm to track branches.

*Will update README soon*