# Lección 9-4: Transformar expresiones exponenciales
Esta sección se enfoca en escribir expresiones exponenciales equivalentes.

Para verificar y comparar distintas expresiones exponenciales, recuerda la expresión
general:<br>
$\qquad y = a \cdot b^{(x-c)} + d.$<br>

Para comparar distintas expresiones, podemos cambiar todos los parámetros.
Aquí hay un ejemplo para comparar tres funciones distintas:<br>
$\qquad a = a_1, a_2, a_3.$<br>
$\qquad b = b_1, b_2, b_3.$<br>
$\qquad c = c_1, c_2, c_3.$<br>
$\qquad d = d_1, d_2, d_3.$<br>

Una aplicación importante surge al considerar distintas tasas de interés.<br>
Considera una cuenta de ahorros con distintas tasas.<br>
Supongamos lo siguiente:<br>
$\qquad a \quad \text{cantidad inicial en dólares.}$<br>
$\qquad r \quad \text{es la tasa de interés anual.}$<br>
$\qquad t \quad \text{es el tiempo en años.}$<br>
$\qquad A \quad \text{es el capital acumulado.}$<br>
Entonces, la cantidad acumulada a lo largo del tiempo está dada por:<br>
$\qquad A(t) = a \cdot (1+r)^t$ &nbsp; donde &nbsp; $t$ &nbsp; es el tiempo en años.<br>

En este ejemplo, tenemos las siguientes sustituciones en comparación con la forma original:<br>
$\qquad A$ &nbsp; reemplaza a &nbsp; $y$.<br>
$\qquad t$ &nbsp; reemplaza a &nbsp; $x$ &nbsp; como la variable principal.<br>
$\qquad 1+r$ &nbsp; reemplaza a &nbsp; $b$ &nbsp; como parámetro que se ajusta.<br>
$\qquad c=d=0$ &nbsp; son fijos.<br>

Para observar el crecimiento mes a mes, tenemos que ajustar la ecuación así:<br>
$\qquad A(t) = a \cdot \left( (1+r)^{1/12} \right)^{12 t}$<br>
En este ejemplo, &nbsp; $(1+r)^{1/12}$ &nbsp; calcula el crecimiento mensual que corresponde a un crecimiento anual de $(1+r)$.

### Ejemplo 9-4.1: Comparar distintas tasas anuales.
Empezando con una cantidad inicial de $a=1000$ dólares, considera
distintas tasas de interés anuales dadas por:<br>
$\qquad r =  1, 2, 5, 10$ &nbsp; por ciento.<br>
¿Cuánto dinero se acumula después de $1, 2, 5$ años?<br>

Puedes graficar las distintas ecuaciones usando 
`Comparar varias funciones en la misma gráfica` de 
[Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).<br>

Si conviertes las ecuaciones a la forma estándar $\qquad y = a \cdot b^{(x-c)} + d$, $\qquad$
puedes crear un video animado bonito que muestre el crecimiento usando
`Animar gráficas exponenciales` de 
[Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).<br>
