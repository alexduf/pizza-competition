def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_neighbour(
        position: tuple[int, int],
        clients: list[dict[str, any]],
        available_space: int
) -> dict[str, any] | None:
    nearest = None
    for client in clients:
        if (client["pizzas"] <= available_space and
                (nearest is None or
                 distance(position, client["position"]) < distance(position, nearest["position"]))
        ):
            nearest = client
    return nearest


def solve_v2(clients: list[dict[str, any]]) -> str:
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
        if not neighbour:
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
