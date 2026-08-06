
# Lesson 9-6: Recursive Formulas
## 9-6.1. Recursive Formula for an Arithmetic Sequence
* Recursive form: given $a_1$ and $a_n = a_{n-1} + d$ for $n \geq 2$.
* Explicit form: $a_n = a_1 + (n-1)\cdot d$ for $n \geq 2$.
* Determine $a_n$ for $n \geq 2$.
* Plot an arithmetic sequence.
* Generate terms and verify that a sequence is arithmetic by computer.

### Example 9-6.1: Generate arithmetic sequence.
Suppose that $a_1=5 \quad\text{and}\quad a_n=a_{n-1} + 2$<br>
Determine $a_2, a_3, a_4, a_5$ using pencil and paper.<br>

Easy verification (minimal coding): Verify your answer using X.1 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

Coding verification (using a for loop to implement the recursive formula): Verify your answer using X.5 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

### Example 9-6.2: Verify arithmetic sequence.
Is $1, 2, 3, 5$ an arithmetic sequence?<br>
Verify your answer using X.2 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).

### Example 1.3: Generate the n-th term of an arithmetic sequence.
Compute $a_100$ for $a_n = 5 + (n-1)\cdot 9$.

Verify your answer using X.6 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

## 9-6.2. Recursive Formula for a geometric sequence
* Recursive form: given $a_1$ and $a_n = r \cdot a_{n-1}$ for $n \geq 2$.
* Explicit form: $a_n = a_1\cdot r^{n-1}$ for $n \geq 2$.
* Determine $a_n$ for $n \geq 2$.
* Plot a geometric sequence.
* Generate terms and verify that a sequence is geometric by computer.

### Example 9-6.3: Generate geometric sequence.
Suppose that $a_1=7 \quad\text{and}\quad a_n= 3 a_{n-1}$<br>
Determine $a_2, a_3, a_4, a_5$ using pencil and paper.<br>
Plot the sequence.<br>
If it is an arithmetic sequence, determine $a_1$ and the recursive formula.<br>

Easy verification (minimal coding): Verify your answer using X.3 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

Coding verification (using a for loop to implement the recursive formula): Verify your answer using X.7 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

### Example 9-6.4: Verify geometric sequence.
Is $1, 2, 4, 8, 16$ an arithmetic sequence?<br>
Plot the sequence.<br>
If it is a geometric sequence, determine $a_1$ and the recursive formula.<br>

Verify your answer using X.4 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).


### Example 9-6.5: Generate the n-th term of an arithmetic sequence.
Compute $a_1=2$ for $a_n = 3 \cdot 12^{n-1}$.

Verify your answer using X.8 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

## Recursive formula for more general problems
* Consider the more general form: $\quad a_n = r\cdot a_{n-1} + d$.
* Apply the more general form to generate terms.

### Example 9-6.6: General formula for geometric sequence.
Let $r=2$, $d=5$, $a_1=3$.<br>
Generate the first five terms of $\quad a_n = r\cdot a_{n-1} + d$.<br>

Verify your answer using X.9 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

