# Lessons 9-5 and 9-6.

# Lesson 9-5: Geometric Sequences
A geometric sequence is a pattern of numbers that begins with a nonzero term
and each term after is found by multiplying the previous term by a nonzero constant $r$. 
The constant $r$ is called the **common ratio**.

Example from the definition:  
For $r=3$, starting from $2$, we have the following geometric sequence:<br>
First term is $2$.<br>
Second term is $2\cdot 3=6$.<br>
Third term is $6\cdot 3=18$.<br>

Geometric sequences are exponential functions.<br>
Let $n$ be a positive integer.<br>
Let the first term be $a_1$, the n-th term be $a_n$, and the common ratio be $r \not= 0$.<br>
The $n$-th term is given by: $\quad a_n=a_1 \cdot r^{n-1}$.

### Example 9-5.1: Verify geometric sequence (Example 1 from section 9.5 of the book).
Is $\quad -432, 144, -48, 16 \quad$ a geometric sequence?<br>
If it is a geometric sequence, determine the **common ratio**.<br>

Verify your answer using X.4 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).

### Example 9-5.2: Find terms of a geometric sequence (Example 3.a. from section 9.5 of the book).
Find the next three terms in $64, 16, 4, 1$.<br>
Step 1. Find the common ratio.<br>
$\qquad \frac{16}{64} = ?,\quad \frac{4}{16} = ?,\quad \frac{1}{4} = ?$.<br>
The common ratio is $r=?$.<br>

Step 2. Multiply by the common ratio.<br>
$\qquad 1 \cdot \frac{1}{4} = ?, \quad 
 \frac{1}{4}  \cdot \frac{1}{4} = ?, \quad 
 \frac{1}{16} \cdot \frac{1}{4} = ?$.<br>
The next three terms are $?, ?, ?$.<br>

Verify your answer using X.4 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).

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
Compute $a_100$ for $a_n = 3 \cdot 12^{n-1}$.

Verify your answer using X.8 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

## Recursive formula for more general problems
* Consider the more general form: $\quad a_n = r\cdot a_{n-1} + d$.

### Example 9-6.6: General formula for geometric sequence.
Let $r=2$, $d=5$, $a_1=3$.<br>
Generate the first five terms of $\quad a_n = r\cdot a_{n-1} + d$.<br>

Verify your answer using X.9 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

