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
$\qquad A \quad \text{refers to the accumulated capital.}$<br>
We then have that the amount accumulated over time is given by:<br>
$\qquad A(t) = a \cdot (1+r)^t$ &nbsp; where &nbsp; $t$ &nbsp; is time in years.<br>

In this example, we have the following substitutions compared to the original form:<br>
$\qquad A$ &nbsp; replaces &nbsp; $y$.<br>
$\qquad t$ &nbsp; replaces &nbsp; $x$ &nbsp; as the main variable.<br>
$\qquad 1+r$ &nbsp; replaces &nbsp; $b$ &nbsp; as a parameter to adjust.<br>
$\qquad c=d=0$ &nbsp; are fixed.<br>

In order to observe the growth monthly, we need to adjust the equation using:<br>
$\qquad A(t) = a \cdot \left( (1+r)^{1/12} \right)^{12 t}$<br>
In this example, &nbsp; $(1+r)^{1/12}$ &nbsp; calculates the monthly rate growth for an annual rate growth of $(1+r)$.

### Example 9-4.1: Compare different annual rates.
Starting from an initial amount of $a=1000$ dollars, consider
different annual percentage rates given by:<br>
$\qquad r =  1, 2, 5, 10$ &nbsp; percent.<br>
What is the accumulated amount of money after $1, 2, 5$ years?<br>

You can plot the different equations using 
`Compare multiple functions on the same plot` from 
[Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>

If you convert the equations in standard form $\qquad y = a \cdot b^{(x-c)} + d$, $\qquad$
you can create a nice animation video that shows the growth using
`Animate exponential plots` from 
[Homework_funs](https://colab.research.google.com/drive/1voxqdIaLmYPqHsSuhzidIW3HJ3wbg8ID?usp=sharing).<br>


