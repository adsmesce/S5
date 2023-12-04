import math
import random

def hill_climbing(objective_function, initial_solution, move_operator, max_iterations):
    current_solution = initial_solution
    current_score = objective_function(current_solution)
    solution_path = [current_solution] 

    for i in range(max_iterations):
        new_solution = move_operator(current_solution)
        new_score = objective_function(new_solution)
        if new_score > current_score:
            current_solution = new_solution
            current_score = new_score
            solution_path.append(current_solution) 

    return current_solution, current_score, solution_path


def objective_function(x):
    return math.sin(x)


def move_operator(x):
    return x + random.uniform(-0.1, 0.1)


initial_solution = 1.0
max_iterations = 10000

solution, score, solution_path = hill_climbing(objective_function, initial_solution, move_operator, max_iterations)

print("Solution : ", solution)
print("Score : ", score)
print("Path of solutions : ",solution_path)



