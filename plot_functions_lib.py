
# Plot multiple functions here.
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from sympy.interactive.printing import init_printing
init_printing()

# Animation class
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from IPython.core.debugger import set_trace

# List of contents:
#
# Animate: video animations of functions plots:
# - special support for integer plots.
# - special support for exponential functions.
# - allows you to plot arbitrary functions.
#
# plot_funs: plots multiple functions.
#
# Arithmetic and geometric sequences
# equal_seqs:    equal sequences.
# arithm_gen:    generates arithmetic sequences.
# check_arithm:  checks if a sequence is arithmetic.
# geom_gen:      generates a geometric sequence.
# check_geom:    checks if a sequence is geometric.
# conv_list_gen: generates a mixed sequence.
# plot_seq:      plots a sequence.


# Animate: video animations of functions plots:
# - special support for integer plots.
# - special support for exponential functions.
# - allows you to plot arbitrary functions.
class Animate:
    def __init__(self, scatter_plot=False, keep_all=True):
        """
        Creates animations of functions.

        scatter_plot: if True, then it plots a scatter plot instead of a line plot.

        keep_all: if True, it keeps all the points from 0 to num_of_points.
                  Otherwise, each frame plots num_of_points/num_of_frames points.
        """

        # Keep all points:
        self.keep_all = keep_all

        # Specify scatter plot:
        self.scatter_plot = scatter_plot

    def set_int_x(self, minX, maxX):
      """
      sets an integer range for the x-axis.

        minX: The minimum value for the x-axis.
        maxX: The maximum value for the x-axis.
      """
      if (not isinstance(minX, int)) or (not isinstance(maxX, int)):
        raise ValueError("minX and maxX must be integers.")
      if not minX < maxX:
        raise ValueError("MinX must be less than MaxX.")

      # Set the limits:
      self.minX = minX
      self.maxX = maxX

      # Create a custom range:
      self.X_list = np.array(range(minX, maxX+1))
      self.num_of_frames = len(self.X_list)-1

      # Create the frame index:
      num_of_points = len(self.X_list)
      self.points = np.array(range(0, num_of_points))

      print(self.X_list)
      print(self.points)


    def set_x_range(self, minX, maxX, num_of_points, num_of_frames):
      """
      sets the range for the x-axis.

        minX: The minimum value for the x-axis.
        maxX: The maximum value for the x-axis.
        num_of_points: Number of points for the x-axis.
        num_of_frames: Number of frames for the animation.
      """

      if not minX < maxX:
        raise ValueError("MinX must be less than MaxX.")
      if not num_of_points > 0:
        raise ValueError("num_of_points must be greater than 0.")

      if num_of_frames > num_of_points:
        print(f"num_of_frames={num_of_frames} num_of_points={num_of_points}")
        raise ValueError("num_of_frames must be less than or equal to num_of_points.")

      # Set the limits:
      self.num_of_frames = num_of_frames

      # Limits:
      self.minX = minX
      self.maxX = maxX

      # Create a custom range:
      self.X_list = np.linspace(minX, maxX, num=num_of_points)

      # Create the frame index:
      self.points = (np.floor(np.linspace(0, num_of_points-1, self.num_of_frames+1))).astype(int)



    def set_labels(self, x_label="X-axis", y_label="Y-axis"):
      """
      x_label: The label for the x-axis.
      y_label: The label for the y-axis.
      num_of_points: The number of points to plot on each frame.
      """
      # X and Y labels:
      self.x_label = x_label
      self.y_label = y_label


    def add_exp_functions(self, a_list, b_list, c_list, d_list):
      """
      Adds a list of exponential functions to the animation defined using the lists for:
      a_list: a values.
      b_list: b values.
      c_list: c values.
      d_list: d values.
      """
      if (
          len(a_list) != len(b_list) or
          len(a_list) != len(c_list) or
          len(a_list) != len(d_list)
          ):
        raise ValueError("All lists must be of the same length!")

      if not a_list:
        raise ValueError("a_list must not be empty!")

      # Create a list of functions and names:
      self.functions = []
      self.names = []
      self.var_name = "x"

      x = sp.symbols('x')
      for a, b, c, d in zip(a_list, b_list, c_list, d_list):
        equation_string = f"{a:.3f} * {b:.3f}**(x-{c:.3f}) + {d:.3f}"
        eqn = sp.parse_expr(equation_string)

        self.functions.append(eqn)
        self.names.append(f"y = {a} * {b}^(x-{c}) + {d}")

    def add_funs(self, funs, var_name, names):
      """
      Adds a list functions to the animation plot.
      funs: contains the list of functions.
      common_var: the common variable.
      names: the names of the functions.
      """
      if (
          len(funs) != len(names)
          ):
        raise ValueError("The number of functions and their list of names must be of the same length!")

      if not var_name:
        raise ValueError("You need to provide a common variable for the functions.")

      # Create a list of functions and names:
      self.var_name = var_name
      self.functions = funs
      self.names = names


    def setup_plot(self):
      """ Sets up a plot based on the list values.
      """
      # Create a figure and axis
      self.fig, self.ax = plt.subplots(figsize=(8, 6))
      self.line_list = []

      # Go through the loop:
      for name in self.names:
        line, = self.ax.plot([], [], label=name)
        self.line_list.append(line)


    def init(self):
      """ initialization function for animation.
      """
      if (self.line_list is None):
        raise ValueError("Call setup_plot first. line_list must be set before calling init.")

      self.ax.set_xlim(self.minX, self.maxX)
      self.ax.set_ylim(0, 1)
      return self.line_list

    def update(self, frame):
      """ Update function for animation.
      """
      if (self.line_list is None):
        raise ValueError("Call setup_plot first. line_list must be set before calling update.")

      # set_trace() # Use for debugging your code.

      # Setup the x range
      if (self.keep_all):
        x_min = self.X_list[0]
        min_index = 0
      else:
        x_min = self.X_list[self.points[frame]]
        min_index = frame

      x_max = self.X_list[self.points[frame+1]]
      x_vals = self.X_list[min_index:self.points[frame+1]+1]

      # print(f"frame={frame}, min_index={min_index}, x_min={x_min}, x_max={x_max}")
      # print(f"x_vals={x_vals}")

      # Setup the functions:
      # Define the symbolic variable
      x = sp.symbols(self.var_name)
      y_min_dynamic = float('inf')
      y_max_dynamic = float('-inf')
      for func, line in zip(self.functions, self.line_list):
        # Convert symbolic function to a numerical function
        numerical_func = sp.lambdify(x, func, 'numpy')
        # Compute y-values
        y = [numerical_func(val) for val in x_vals]
        if self.scatter_plot:
          plt.scatter(x_vals, y)
        else:
          line.set_data(x_vals, y)

        # Update
        y_min_dynamic = min(y_min_dynamic, min(y))
        y_max_dynamic = max(y_max_dynamic, max(y))

      # Update x and y ranges
      x_min_dynamic = x_min # Modify here to maintain the left margin
      x_max_dynamic = x_max

      # Add a margin of 10% for both axes
      x_margin = (x_max_dynamic - x_min_dynamic) * 0.1
      y_margin = (y_max_dynamic - y_min_dynamic) * 0.1

      self.ax.set_xlim(x_min_dynamic - x_margin, x_max_dynamic + x_margin)
      self.ax.set_ylim(y_min_dynamic - y_margin, y_max_dynamic + y_margin)

      # Update title dynamically
      self.ax.set_title(f"Frame {frame}")

      return self.line_list

    def create_animation(self, x_label="x axis", y_label="y axis", interval=500):
      """ Creates the animation given:
      x_label: The label for the x-axis.
      y_label: The label for the y-axis.
      interval: The interval between frames in milliseconds.
      """
      self.ani = FuncAnimation(self.fig, self.update, frames=self.num_of_frames, init_func=self.init, interval=interval, blit=False)

      # Add labels, legend, and grid
      self.ax.set_xlabel(x_label)
      self.ax.set_ylabel(y_label)
      self.ax.legend()
      self.ax.grid(True)

      # Show the animation
      self.fig.show()

      return(self.ani)

# Single function to plot multiple functions.
def plot_funs(functions,
              t_values,
              names,
              var_name="t",
              title="Comparison of multiple functions",
              x_label="t",
              y_label="y"):
    """
    Plots multiple sequences based on given equations.

    Parameters:
    - functions: List of symbolic equations to plot.
    - t_values: List or array of integer t-values shared across all functions.
    - names: List of names corresponding to each function.
    - title: Title of the plot (default: "Symbolic Sequences Plot").
    - x_label: Label for the x-axis (default: "t").
    - y_label: Label for the y-axis (default: "y").
    """

    # set_trace() # Use for debugging your code.

    if len(functions) != len(names):
      raise ValueError("The number of functions and names must be the same!")

    if not functions:
        raise ValueError("functions must not be empty!")

    # Define the symbolic variable
    x = sp.symbols(var_name)

    # Ensure the lengths of functions and names match
    if len(functions) != len(names):
        raise ValueError("The number of functions and names must be the same.")

    # Initialize the plot
    plt.figure(figsize=(8, 6))

    # Colors and markers for variety
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', '*']

    # Loop through each function and plot it
    for i, (func, name) in enumerate(zip(functions, names)):
        # Convert symbolic function to a numerical function
        numerical_func = sp.lambdify(x, func, 'numpy')
        # Compute y-values
        y_values = [numerical_func(val) for val in t_values]
        # Plot with distinct color and marker
        func_str = str(func.evalf(5)) 
        plt.plot(t_values, y_values, label=f"{name}: {func_str}",
                 color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle='-')

    # Add labels, legend, and title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(0, color='black', linewidth=0.8, linestyle="--")  # x-axis
    plt.axvline(0, color='black', linewidth=0.8, linestyle="--")  # y-axis
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid
    plt.legend(fontsize=12)

    # Show the plot
    plt.show()

# Arithmetic and geometric sequences
# equal_seqs:    equal sequences.
# arithm_gen:    generates arithmetic sequences.
# check_arithm:  checks if a sequence is arithmetic.
# geom_gen:      generates a geometric sequence.
# check_geom:    checks if a sequence is geometric.
# conv_list_gen: generates a mixed sequence.
# plot_seq:      plots a sequence.

def equal_seqs(seq1:list[float], seq2:list[float]) -> bool:
  """ returns true if two squences are the same.

  seq1: first sequence
  seq2: second sequence
  """
  if len(seq1) != len(seq2):
    my_len = min(len(seq1), len(seq2))
    print("Sequences are of different lengths:")
    print(f"Length of sequence 1: {len(seq1)}")
    print(f"Length of sequence 2: {len(seq2)}")
    print(f"Only comparing {my_len} numbers.")

  return (seq1[0:my_len] == seq2[0:my_len])


def arithm_gen(first_num:float, d:float, n_range:int) -> list[int]:
  """ returns a list of the first n_range terms of the arithmetic sequence.
        an = a1 + (n-1)d.

  first_num: the first number in the sequence
  d: the common difference
  n_range: the number of terms in the sequence
  """
  a = first_num
  a_list = []
  for n in range(n_range):
    a_list.append(a)
    a = a + d
  return a_list


def check_arithm(a_values:list[float]) -> bool:
  """ returns True if the sequence is an arithmetic sequence, False otherwise.

  a_values: list of values for the sequence.
  """
  # Must have at-least two values:
  if len(a_values) < 3:
    return False

  # Form the difference list:
  d = a_values[1] - a_values[0]
  d_values = np.array(a_values[2:]) - np.array(a_values[1:len(a_values)-1])
  return all(d_values == d)




def geom_gen(first_num:float, r:float, n_range:int) -> list[int]:
  """ returns a list of the first n_range terms of the geometric sequence.
              an = r^n * a1

  first_num: the first number in the sequence
  r: the common ratio
  n_range: the number of terms in the sequence
  """
  a = first_num
  a_list = []
  for n in range(n_range):
    a_list.append(a)
    a = r*a
  return a_list


def check_geom(a_values:list[float]) -> bool:
  """ returns True if the sequence is a geometic sequence, False otherwise.

  a_values: list of values for the sequence.
  """
  # Must have at-least two values:
  if len(a_values) < 3:
    return False

  # Form the difference list:
  r = a_values[1] / a_values[0]
  d_values = np.array(a_values[2:]) / np.array(a_values[1:len(a_values)-1])
  return all(d_values == r)


def conv_list_gen(first_num:float, d:float, r:float, n_range:int) -> list[int]:
  """
      returns a list of the first n_range terms of the mixed sequence:
              a1  = first_num
              a2  = a1 + d
              a3  = a1 + (a2 - a1) / r
              a4  = a1 + (a3 - a2) / r
              :
              an  = a1 + (a_(n-1) - a_(n-2)) / r

  first_num: the first number in the sequence
  d: the common difference
  r: the common ratio
  """
  a_prev_prev = first_num  # a_(n-2)
  a_prev = first_num + d   # a_(n-1)
  a_list = [a_prev_prev, a_prev]

  for n in range(n_range-2):
    a = a1 + (a_prev - a_prev_prev) / r
    a_list.append(a)

    # Update:
    a_prev_prev = a_prev
    a_prev = a
  return a_list


def plot_seq(a_values:list[float], title="Sequence Plot", x_label=r"n", y_label=r"$a_n$"):
    """
    Plots a sequence stored in a_values.

    Parameters:
    a_values (list): A list of values representing the sequence.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    """
    if not a_values:
        print("The list of a-values is empty. Nothing to plot.")
        return

    # Generate x-coordinates as integers from 1 to len(y_values)
    x = list(range(1, len(a_values) + 1))

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, a_values, color='blue', marker='o', label=r'$a_n$')
    # plt.plot(x, a_values, color='blue', linestyle='-', alpha=0.7)  # Connect points with a line

    # Add labels and title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(0, color='black', linewidth=0.8, linestyle="--")  # Horizontal line at y=0
    plt.axvline(0, color='black', linewidth=0.8, linestyle="--")  # Vertical line at x=0
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid
    plt.legend()

    # Show the plot
    plt.show()



