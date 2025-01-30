def solve_v1(clients: list[dict[str, any]]) -> str:
    tours: list[list[int]] = [[client["id"]] for client in clients]

    tours_string = ""
    for tour in tours:
        tours_string += " ".join(map(str, tour)) + "\n"

    return tours_string
