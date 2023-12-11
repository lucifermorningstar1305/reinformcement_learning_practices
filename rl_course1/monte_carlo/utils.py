from typing import Dict, List, Callable
import numpy as np


def print_policy(policy: Dict, grid: Callable):
    """Function to plot the policy of the grid world"""

    for row in range(grid.nrows):
        print("-" * 10)
        for col in range(grid.ncols):
            act = policy[(row, col)] if not grid.is_terminal((row, col)) else " "
            print(f" {act} |", end="")

        print("")


def print_values(values: Dict, grid: Callable):
    """Function to print the values of the states of the grid world"""

    for row in range(grid.nrows):
        print("-" * 10)
        for col in range(grid.ncols):
            v = values.get((row, col), 0.0)
            if v >= 0:
                print(f" {v:.2f}|", end="")
            else:
                print(f"{v:.2f}|", end="")

        print("")
