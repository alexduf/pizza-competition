# Données initiales
from solver_v1 import solve_v1
from solver_v2 import solve_v2

depot = (0, 0)  # Position du dépôt
capacity = 10  # Capacité maximale du scooter

# Distance de Manhattan
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def load_clients(file_path: str) -> list[dict[str, any]]:
    clients = []
    with open(file_path, "r") as fi:
        for line in fi.readlines()[1:]:  # Ignorer l'en-tête
            client_id, x, y, pizzas = map(int, line.strip().split(","))
            clients.append({
                "id": client_id,
                "position": (x, y),
                "pizzas": pizzas
            })
    return clients


def get_score(tours_string: str):
    # Renvoie le score, la validité et un message d'erreur éventuel
    if not isinstance(tours_string, str):
        return 0, False, "❌ Erreur : Les tournées doivent être une chaîne de caractères"
    
    try:
        tours = [list(map(int, tour.split())) for tour in tours_string.strip().split("\n")]
    except ValueError:
        return 0, False, "❌ Erreur : Les tournées doivent être des entiers séparés par des espaces."
    except Exception as e:
        print(e)
        return 0, False, "❌ Erreur : Une erreur inattendue s'est produite."

    
    clients = load_clients("dataset.csv")
    
    client_ids = {client["id"] for client in clients}
    delivered_ids = set()
    total_distance = 0
    
    for tour in tours:
        current_load = 0
        current_position = depot
        for client_id in tour:
            client = next((c for c in clients if c["id"] == client_id), None)
            if not client:
                return 0, False, f"❌ Erreur : Le client {client_id} n'existe pas."
            if client_id in delivered_ids:
                return 0, False, f"❌ Erreur : Le client {client_id} est livré plusieurs fois."
            current_load += client["pizzas"]
            if current_load > capacity:
                return 0, False, f"❌ Erreur : Une tournée dépasse la capacité maximale de {capacity} pizzas."
            
            delivered_ids.add(client_id)
            total_distance += manhattan_distance(current_position, client["position"])    
            current_position = client["position"]    

    if delivered_ids != client_ids:
        return 0, False, f"❌ Erreur : {len(client_ids - delivered_ids)} clients n'ont pas été livrés."

    return total_distance, True, "✅ Solution valide."

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
        file_name = f'sol_{score}_{date}'

        with open(f'solutions/{file_name}.txt', 'w') as f:
            f.write(tours)
        print('Solution sauvegardée')
        if draw:
            plot_solution(tours)
    return tours

if __name__ == "__main__":
    import datetime
    clients = load_clients("dataset.csv") # les clients sont sockés dans une liste de dict, avec pour clé "id", "position", "pizzas"
    evaluate_solver(solve_v1, clients, draw=False)
    evaluate_solver(solve_v2, clients, draw=True)
