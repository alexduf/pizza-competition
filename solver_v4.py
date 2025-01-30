

from evaluator import manhattan_distance


def nearest_neighbour(
        position: tuple[int, int],
        clients: list[dict[str, any]],
        available_space: int
) -> dict[str, any] | None:
    nearest = None
    for client in clients:
        if (client["pizzas"] <= available_space and
                (nearest is None or
                 manhattan_distance(position, client["position"]) < manhattan_distance(position, nearest["position"]))
        ):
            nearest = client
    return nearest


def farthest_neighbour(
        position: tuple[int, int],
        clients: list[dict[str, any]],
        available_space: int
) -> dict[str, any] | None:
    farthest = None
    for client in clients:
        if (client["pizzas"] <= available_space and
                (farthest is None or
                 manhattan_distance(position, client["position"]) > manhattan_distance(position, farthest["position"]))
        ):
            farthest = client
    return farthest

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
            neighbour = farthest_neighbour(current_position, clients, available_space)
        else:
            neighbour = nearest_neighbour(current_position, clients, available_space)
        if not neighbour or manhattan_distance(current_position, neighbour["position"]) > manhattan_distance((0, 0), neighbour["position"]):
            return current_tour
        else:
            current_position = neighbour["position"]
            available_space -= neighbour["pizzas"]
            clients.remove(neighbour)
            current_tour.append(neighbour["id"])
    return current_tour

def solve_v4(clients: list[dict[str, any]]) -> str:
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
