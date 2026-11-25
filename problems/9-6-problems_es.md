# Lección 9-6: Fórmulas recursivas
## 9-6.1. Fórmula recursiva de una sucesión aritmética
* Forma recursiva: dados $a_1$ y $a_n = a_{n-1} + d$ para $n \geq 2$.
* Forma explícita: $a_n = a_1 + (n-1)\cdot d$ para $n \geq 2$.
* Determinar $a_n$ para $n \geq 2$.
* Graficar una sucesión aritmética.
* Generar términos y verificar con la computadora que una sucesión es aritmética.

### Ejemplo 9-6.1: Generar una sucesión aritmética.
Supongamos que $a_1=5 \quad\text{y}\quad a_n=a_{n-1} + 2$<br>
Determina $a_2, a_3, a_4, a_5$ con lápiz y papel.<br>

Verificación fácil (con muy poco código): verifica tu respuesta con X.1 de [Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).<br>

Verificación con programación (usando un ciclo for para aplicar la fórmula recursiva): verifica tu respuesta con X.5 de [Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).<br>

### Ejemplo 9-6.2: Verificar una sucesión aritmética.
¿Es $1, 2, 3, 5$ una sucesión aritmética?<br>
Verifica tu respuesta con X.2 de [Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).

### Ejemplo 1.3: Generar el n-ésimo término de una sucesión aritmética.
Calcula $a_100$ para $a_n = 5 + (n-1)\cdot 9$.

Verifica tu respuesta con X.6 de [Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).<br>

## 9-6.2. Fórmula recursiva de una sucesión geométrica
* Forma recursiva: dados $a_1$ y $a_n = r \cdot a_{n-1}$ para $n \geq 2$.
* Forma explícita: $a_n = a_1\cdot r^{n-1}$ para $n \geq 2$.
* Determinar $a_n$ para $n \geq 2$.
* Graficar una sucesión geométrica.
* Generar términos y verificar con la computadora que una sucesión es geométrica.

### Ejemplo 9-6.3: Generar una sucesión geométrica.
Supongamos que $a_1=7 \quad\text{y}\quad a_n= 3 a_{n-1}$<br>
Determina $a_2, a_3, a_4, a_5$ con lápiz y papel.<br>
Grafica la sucesión.<br>
Si es una sucesión aritmética, determina $a_1$ y la fórmula recursiva.<br>

Verificación fácil (con muy poco código): verifica tu respuesta con X.3 de [Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).<br>

Verificación con programación (usando un ciclo for para aplicar la fórmula recursiva): verifica tu respuesta con X.7 de [Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).<br>

### Ejemplo 9-6.4: Verificar una sucesión geométrica.
¿Es $1, 2, 4, 8, 16$ una sucesión aritmética?<br>
Grafica la sucesión.<br>
Si es una sucesión geométrica, determina $a_1$ y la fórmula recursiva.<br>

Verifica tu respuesta con X.4 de [Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).


### Ejemplo 9-6.5: Generar el n-ésimo término de una sucesión aritmética.
Calcula $a_1=2$ para $a_n = 3 \cdot 12^{n-1}$.

Verifica tu respuesta con X.8 de [Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).<br>

## Fórmula recursiva para problemas más generales
* Considera la forma más general: $\quad a_n = r\cdot a_{n-1} + d$.
* Aplica la forma más general para generar términos.

### Ejemplo 9-6.6: Fórmula general de una sucesión geométrica.
Sean $r=2$, $d=5$, $a_1=3$.<br>
Genera los primeros cinco términos de $\quad a_n = r\cdot a_{n-1} + d$.<br>

Verifica tu respuesta con X.9 de [Homework_funs](https://colab.research.google.com/github/pattichis/GraphFuns2/blob/main/Homework_funs_es.ipynb).<br>
