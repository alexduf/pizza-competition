

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


def solve_v3(clients: list[dict[str, any]]) -> str:
    tours: list[list[int]] = []
    available_space = 10
    current_position = (0, 0)
    current_tour = []

    while len(clients) > 0:
        if available_space <= 0:
            available_space = 10
            current_position = (0, 0)
            tours.append(current_tour)
            current_tour = []

        neighbour = nearest_neighbour(current_position, clients, available_space)
        if not neighbour or manhattan_distance(current_position, neighbour["position"]) > manhattan_distance((0, 0), neighbour["position"]):
            available_space = 10
            current_position = (0, 0)
            tours.append(current_tour)
            current_tour = []
        else:
            current_position = neighbour["position"]
            available_space -= neighbour["pizzas"]

            clients.remove(neighbour)
            current_tour.append(neighbour["id"])

    if len(current_tour) > 0:
        tours.append(current_tour)

    tours_string = ""
    for tour in tours:
        tours_string += " ".join(map(str, tour)) + "\n"

    return tours_string
