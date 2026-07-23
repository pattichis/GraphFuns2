# Lesson 9-1: Exponential Functions


## 1. Exponential growth functions
> * ** General form: ** $f(x) = a \cdot b^x$, \quad a>0, b>1$.
> * ** Domain: ** all real numbers.
> * ** Range:  ** positive real numbers.
> * ** Intercepts: ** one y-intercept, no x-intercepts.
> * ** As x increases: ** $f(x)$ increases.
> * ** As x decreases: ** $f(x)$ approaches zero.

### Problem Statement
Prove that for any positive integer $n$, the sum of the first $n$ odd integers is $n^2$.

> **Given:** $S_n = 1 + 3 + 5 + \dots + (2n - 1)$

---

### Solution / Proof Sketch

#### Base Case ($n = 1$)
$$S_1 = 2(1) - 1 = 1 = 1^2$$

#### Inductive Step
Assume $S_k = k^2$ holds true for $n = k$. We must show $S_{k+1} = (k+1)^2$:

1. $S_{k+1} = S_k + (2k + 1)$
2. $S_{k+1} = k^2 + 2k + 1$ *(by induction hypothesis)*
3. $S_{k+1} = (k + 1)^2$ $\quad\blacksquare$
