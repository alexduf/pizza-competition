import random
import math
from evaluator import get_score

def solve_v7(clients):
    print("\n🐜 Starting Ant Colony Optimization Solver (v7)...")
    print("🚀 Initializing parameters and pheromones...")

    depot = (0, 0)  # Assuming depot is at (0, 0)
    num_ants = 10
    num_iterations = 100
    alpha = 1  # Pheromone importance
    beta = 2  # Distance importance
    evaporation_rate = 0.5
    Q = 100  # Pheromone deposit factor

    num_clients = len(clients)
    distances = [[math.dist(clients[i]['position'], clients[j]['position']) for j in range(num_clients)] for i in range(num_clients)]
    pheromones = [[1 for _ in range(num_clients)] for _ in range(num_clients)]

    def calculate_probabilities(ant, visited):
        probabilities = []
        current = ant[-1]
        for next_client in range(num_clients):
            if next_client not in visited:
                tau = pheromones[current][next_client] ** alpha
                eta = (1 / distances[current][next_client]) ** beta if distances[current][next_client] > 0 else 0
                probabilities.append(tau * eta)
            else:
                probabilities.append(0)
        total = sum(probabilities)
        return [p / total if total > 0 else 0 for p in probabilities]

    def construct_solution():
        ants = [[random.randint(0, num_clients - 1)] for _ in range(num_ants)]
        for ant in ants:
            visited = set(ant)
            while len(visited) < num_clients:
                probabilities = calculate_probabilities(ant, visited)
                next_client = random.choices(range(num_clients), weights=probabilities)[0]
                ant.append(next_client)
                visited.add(next_client)
        return ants

    def update_pheromones(solutions):
        nonlocal pheromones
        for i in range(num_clients):
            for j in range(num_clients):
                pheromones[i][j] *= (1 - evaporation_rate)
        for solution in solutions:
            score = 1 / sum(distances[solution[i]][solution[i + 1]] for i in range(len(solution) - 1))
            for i in range(len(solution) - 1):
                pheromones[solution[i]][solution[i + 1]] += Q * score

    # Initialize best_solution with a default valid solution
    best_solution = [[i] for i in range(num_clients)]  # Each client in its own tour
    best_score, valid, message = get_score("\n".join(" ".join(str(clients[client]['id']) for client in best_solution)))
    if not valid:
        print(f"❌ Initial default solution is invalid: {message}")
        return ""

    for iteration in range(num_iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{num_iterations}...")
        solutions = construct_solution()
        for solution in solutions:
            tours_string = "\n".join(" ".join(str(clients[client]['id']) for client in solution))
            score, valid, message = get_score(tours_string)
            if valid and score < best_score:
                best_score = score
                best_solution = solution
                print(f"✨ New best score: {best_score}")
        update_pheromones(solutions)

    print("\n✅ Optimization complete!")

    # Use the get_score function from evaluator.py to validate the solution
    tours_string = "\n".join(" ".join(str(client) for client in tour) for tour in best_solution)
    score, valid, message = get_score(tours_string)
    if valid:
        print(f"🏆 Best score achieved: {score}")
    else:
        print(f"❌ Invalid solution: {message}")

    return tours_string