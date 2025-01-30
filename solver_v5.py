import random

from evaluator import manhattan_distance, get_score


def next_delivery_candidates(
        position: tuple[int, int],
        clients: list[dict[str, any]],
        available_space: int
) -> list[dict[str, any]]:
    clients = clients.copy()
    # sort clients by distance
    clients.sort(key=lambda client: manhattan_distance(position, client["position"]))
    # filter such that we keep only if we have the capacity
    clients = list(filter(lambda client: client["pizzas"] <= available_space, clients))
    # first 10 results
    return clients[:2]

def first_delivery_candidates(
        position: tuple[int, int],
        clients: list[dict[str, any]],
        available_space: int
) -> list[dict[str, any]]:
    clients = clients.copy()
    # sort clients by distance
    clients.sort(key=lambda client: manhattan_distance(position, client["position"]), reverse=True)
    # filter such that we keep only if we have the capacity
    # clients = list(filter(lambda client: client["pizzas"] <= available_space, clients))
    # first 10 results
    return clients[:5]

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

def solve_v5(clients: list[dict[str, any]]) -> str:
    min_score = 999999
    min_tour = ""

    for i in range(10000):
        tours_string = plan_tours(clients)
        score, valid, message = get_score(tours_string)
        if score < min_score:
            min_score = score
            min_tour = tours_string
            print(f"New best score {min_score}")


    return min_tour
