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

### Example 9-5.1: Verify geometric sequence (Example 1 from section 9.5 of the book).
Is $-432, 144, -48, 16$ a geometric sequence?<br>
If it is a geometric sequence, determine the **common ratio**.<br>

Verify your answer using X.4 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).

### Example 9-5.2: Find terms of a geometric sequence (Example 3.a. from section 9.5 of the book).
Find the next three terms in $64, 16, 4, 1$.<br>
Step 1. Find the common ratio.<br>
$\frac{16}{64} = ?, \frac{4}{16} = ?, \frac{1}{4} = ?$.<br>
The common ratio is $r=?$.<br>

Step 2. Multiply by the common ratio.<br>
$1 \cdot \frac{1}{4} = ?, 
 \frac{1}{4}  \cdot \frac{1}{4} = ?, 
 \frac{1}{16} \cdot \frac{1}{4} = ?$.<br>
The next three terms are $?, ?, ?$.<br>

Verify your answer using X.4 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).




# Lesson 9-6: Recursive Formulas
## 1. Recursive Formula for an Arithmetic Sequence
* Recursive form: given $a_1$ and $a_n = a_{n-1} + d$ for $n \geq 2$.
* Explicit form: $a_n = a_1 + (n-1)\cdot d$ for $n \geq 2$.
* Determine $a_n$ for $n \geq 2$.
* Plot an arithmetic sequence.
* Generate terms and verify that a sequence is arithmetic by computer.

### Example 1.1: Generate arithmetic sequence.
Suppose that $a_1=5 \quad\text{and}\quad a_n=a_{n-1} + 2$<br>
Determine $a_2, a_3, a_4, a_5$ using pencil and paper.<br>

Easy verification (minimal coding): Verify your answer using X.1 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

Coding verification (using a for loop to implement the recursive formula): Verify your answer using X.5 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

### Example 1.2: Verify arithmetic sequence.
Is $1, 2, 3, 5$ an arithmetic sequence?<br>
Verify your answer using X.2 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).

### Example 1.3: Generate the n-th term of an arithmetic sequence.
Compute $a_100$ for $a_n = 5 + (n-1)\cdot 9$.

Verify your answer using X.6 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

## 2. Recursive Formula for a geometric sequence
* Recursive form: given $a_1$ and $a_n = r \cdot a_{n-1}$ for $n \geq 2$.
* Explicit form: $a_n = a_1\cdot r^{n-1}$ for $n \geq 2$.
* Determine $a_n$ for $n \geq 2$.
* Plot a geometric sequence.
* Generate terms and verify that a sequence is geometric by computer.

### Example 2.1: Generate geometric sequence.
Suppose that $a_1=7 \quad\text{and}\quad a_n= 3 a_{n-1}$<br>
Determine $a_2, a_3, a_4, a_5$ using pencil and paper.<br>
Plot the sequence.<br>
If it is an arithmetic sequence, determine $a_1$ and the recursive formula.<br>

Easy verification (minimal coding): Verify your answer using X.3 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

Coding verification (using a for loop to implement the recursive formula): Verify your answer using X.7 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

### Example 2.2: Verify geometric sequence.
Is $1, 2, 4, 8, 16$ an arithmetic sequence?<br>
Plot the sequence.<br>
If it is a geometric sequence, determine $a_1$ and the recursive formula.<br>

Verify your answer using X.4 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).


### Example 2.3: Generate the n-th term of an arithmetic sequence.
Compute $a_100$ for $a_n = 3 \cdot 12^{n-1}$.

Verify your answer using X.8 from [Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>


