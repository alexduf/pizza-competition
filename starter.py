# Données initiales
from evaluator import get_score, load_clients, depot
from solver_v1 import solve_v1
from solver_v2 import solve_v2
from solver_v3 import solve_v3
from solver_v4 import solve_v4
from solver_v5 import solve_v5
from solver_v6 import solve_v6


def plot_solution(tours_string: str):
    import matplotlib.pyplot as plt
    import numpy as np

    tours = [list(map(int, tour.split())) for tour in tours_string.strip().split("\n")]
    clients = load_clients("dataset.csv")

    for tour in tours:
        x = [depot[0]] + [clients[client_id - 1]["position"][0] for client_id in tour] + [depot[0]]
        y = [depot[1]] + [clients[client_id - 1]["position"][1] for client_id in tour] + [depot[1]]
        plt.plot(x, y, marker="o")

    plt.scatter([client["position"][0] for client in clients], [client["position"][1] for client in clients], color="red")
    plt.scatter(depot[0], depot[1], color="green")
    plt.gca().set_aspect("equal")
    plt.show()

def evaluate_solver(
        solver: lambda x: str,
        clients: list[dict[str, any]],
        draw: bool = False
) -> tuple[int, bool, str]:
    tours = solver(clients)
    score, valid, message = get_score(tours)
    if valid:
        print(f"Score : {score}")

        date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'sol_{score}_{solver.__name__}_{date}'

        with open(f'solutions/{file_name}.txt', 'w') as f:
            f.write(tours)
        print('Solution sauvegardée')
        if draw:
            plot_solution(tours)
    return tours

if __name__ == "__main__":
    import datetime
    clients = load_clients("dataset.csv") # les clients sont sockés dans une liste de dict, avec pour clé "id", "position", "pizzas"
    # evaluate_solver(solve_v1, clients, draw=False)
    # evaluate_solver(solve_v2, clients, draw=False)
    # evaluate_solver(solve_v3, clients, draw=False)
    # evaluate_solver(solve_v4, clients, draw=True)
    # evaluate_solver(solve_v5, clients, draw=False)
    evaluate_solver(solve_v6, clients, draw=False)
