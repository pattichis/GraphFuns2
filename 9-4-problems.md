# Lesson 9-4: Transforming Exponential Expressions
This section is focused on writing equivalent exponential expressions.

In order to verify and compare different exponential expressions, recall the general
expression:<br>
$\qquad y = a \cdot b^{(x-c)} + d.$<br>

To compare different expressions, we can vary all of the parameters.
Here is an example for comparing three different functions:<br>
$\qquad a = a_1, a_2, a_3.$<br>
$\qquad b = b_1, b_2, b_3.$<br>
$\qquad c = c_1, c_2, c_3.$<br>
$\qquad d = d_1, d_2, d_3.$<br>

An important application comes from considering different interest rates.<br>
Consider a savings account with different rates.<br>
Assume the following:<br>
$\qquad a \quad \text{initial amount in US dollars.}$<br>
$\qquad r \quad \text{is the annual percentage rate.}$<br>
$\qquad t \quad \text{refers to the time in years.}$<br>
We then have that the amount accumulated over time is given by:<br>
$\qquad y(t) = a \cdot (1+r)^t$ &nbsp; where &nbsp; $t$ &nbsp; is time in years.<br>

In this example, we have the following substitutions compared to the original form:<br>
$\qquad t$ &nbsp; replaces &nbsp; $x$ &nbsp; as the main variable.<br>
$\qquad 1+r$ &nbsp; replaces &nbsp; $b$ &nbsp; as a parameter to adjust.<br>
$\qquad c=d=0 &nbsp; are fixed.<br>

In order to observe the growth monthly, we need to adjust the equation using:<br>
$\qquad y(t) = a \cdot \left( (1+r)^{1/12} \right)^{12 t}$ &nbsp; where &nbsp; $t$ &nbsp; is time in months.<br>



The basic <br>
$ y = a (1+r)^t $

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
