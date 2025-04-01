import random
from datetime import datetime

from evaluator import manhattan_distance, get_score


def next_delivery_candidates(
        position: tuple[int, int],
        clients: list[dict[str, any]],
        available_space: int
) -> list[dict[str, any]]:
    clients = clients.copy()
    # filter such that we keep only if we have the capacity
    clients = list(filter(lambda client: client["pizzas"] <= available_space, clients))
    # order by distance from current position + point to (0, 0)
    clients.sort(key=lambda client: manhattan_distance(position, client["position"]) + manhattan_distance((0, 0), client["position"]))
    # first result
    return clients[:1]

def first_delivery_candidates(
        position: tuple[int, int],
        clients: list[dict[str, any]],
        available_space: int
) -> list[dict[str, any]]:
    clients = clients.copy()
    # sort clients by distance
    clients.sort(key=lambda client: manhattan_distance(position, client["position"]), reverse=True)

    # first 10 results
    return clients[:15]

def plan_tour(
        clients: list[dict[str, any]]
) -> list[int]:
    available_space = 10
    current_position = (0, 0)
    current_tour = []
    # clone array
    clients = clients.copy()

    while len(clients) > 0:
        if available_space <= 0:
            return current_tour

        if len(current_tour) == 0:
            candidates = first_delivery_candidates(current_position, clients, available_space)
            # pick one random
            neighbour = candidates[random.randint(0, len(candidates) - 1)]
        else:
            candidates = next_delivery_candidates(current_position, clients, available_space)
            if len(candidates) > 0:
                neighbour = candidates[random.randint(0, len(candidates) - 1)]
            else:
                neighbour = None
        if not neighbour or manhattan_distance(current_position, neighbour["position"]) > manhattan_distance((0, 0), neighbour["position"]):
            return current_tour
        else:
            current_position = neighbour["position"]
            available_space -= neighbour["pizzas"]
            clients.remove(neighbour)
            current_tour.append(neighbour["id"])
    return current_tour

def plan_tours(clients: list[dict[str, any]]) -> str:
    tours: list[list[int]] = []

    while len(clients) > 0:
        current_tour = plan_tour(clients)
        tours.append(current_tour)
        for client_id in current_tour:
            clients = [client for client in clients if client["id"] != client_id]

    tours_string = ""
    for tour in tours:
        tours_string += " ".join(map(str, tour)) + "\n"
    return tours_string

def solve_v6(clients: list[dict[str, any]]) -> str:
    min_score = 18108
    best_score = 99999
    min_tour = ""

    for i in range(1000000):
        tours_string = plan_tours(clients)
        score, valid, message = get_score(tours_string)
        if score < best_score:
            best_score = score
            print(f"New best score {best_score}")
        if score < min_score:
            min_score = score
            min_tour = tours_string
            print(f"New absolute best score {min_score}, saving results")
            date = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f'sol_{score}_solve_v5_{date}'

            with open(f'solutions/{file_name}.txt', 'w') as f:
                f.write(tours_string)


    return min_tour
