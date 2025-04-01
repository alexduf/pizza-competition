import random
import math
from datetime import datetime
from copy import deepcopy

from evaluator import manhattan_distance, get_score, depot, capacity

def calculate_tour_distance(tour, clients_dict, return_to_depot=True):
    """Calculate the total distance of a tour."""
    if not tour:
        return 0
    
    distance = 0
    current_pos = depot
    
    for client_id in tour:
        client_pos = clients_dict[client_id]["position"]
        distance += manhattan_distance(current_pos, client_pos)
        current_pos = client_pos
    
    if return_to_depot:
        distance += manhattan_distance(current_pos, depot)
    
    return distance

def calculate_total_distance(tours, clients_dict):
    """Calculate the total distance of all tours."""
    total = 0
    for tour in tours:
        total += calculate_tour_distance(tour, clients_dict)
    return total

def is_valid_tour(tour, clients_dict):
    """Check if a tour is valid (within capacity constraints)."""
    total_pizzas = sum(clients_dict[client_id]["pizzas"] for client_id in tour)
    return total_pizzas <= capacity

def create_initial_solution(clients):
    """Create an initial feasible solution using a greedy approach."""
    clients_dict = {client["id"]: client for client in clients}
    unassigned = list(clients_dict.keys())
    tours = []
    
    while unassigned:
        tour = []
        remaining_capacity = capacity
        current_pos = depot
        
        # Keep adding clients to the current tour until capacity is reached
        while unassigned:
            # Find the nearest unassigned client that fits
            nearest = None
            min_distance = float('inf')
            
            for client_id in unassigned:
                if clients_dict[client_id]["pizzas"] <= remaining_capacity:
                    client_pos = clients_dict[client_id]["position"]
                    dist = manhattan_distance(current_pos, client_pos)
                    if dist < min_distance:
                        min_distance = dist
                        nearest = client_id
            
            if nearest is None:
                break  # No more clients fit in this tour
                
            # Add the nearest client to the tour
            tour.append(nearest)
            unassigned.remove(nearest)
            remaining_capacity -= clients_dict[nearest]["pizzas"]
            current_pos = clients_dict[nearest]["position"]
        
        if tour:
            tours.append(tour)
    
    return tours, clients_dict

def get_neighbor(current_solution, clients_dict, neighborhood_type):
    """Generate a neighbor solution by making a small modification."""
    # Deep copy to avoid modifying the original
    new_solution = deepcopy(current_solution)
    
    if len(new_solution) <= 1:
        return new_solution  # Not enough tours to modify
    
    # Choose randomly from different neighborhood operations
    if neighborhood_type == 0 and random.random() < 0.6:
        # MOVE: Move a random client from one tour to another
        from_tour_idx = random.randint(0, len(new_solution) - 1)
        if not new_solution[from_tour_idx]:
            return new_solution  # Empty tour, can't move from it
            
        to_tour_idx = random.randint(0, len(new_solution) - 1)
        while to_tour_idx == from_tour_idx:
            to_tour_idx = random.randint(0, len(new_solution) - 1)
            
        client_idx = random.randint(0, len(new_solution[from_tour_idx]) - 1)
        client_id = new_solution[from_tour_idx][client_idx]
        
        # Check if moving this client would violate capacity constraints
        new_tour = new_solution[to_tour_idx] + [client_id]
        if is_valid_tour(new_tour, clients_dict):
            new_solution[from_tour_idx].pop(client_idx)
            new_solution[to_tour_idx].append(client_id)
            
            # Remove empty tours
            new_solution = [tour for tour in new_solution if tour]
    
    elif neighborhood_type == 1 or random.random() < 0.5:
        # SWAP: Swap two clients between different tours
        if len(new_solution) < 2:
            return new_solution
            
        tour1_idx = random.randint(0, len(new_solution) - 1)
        tour2_idx = random.randint(0, len(new_solution) - 1)
        
        # Ensure both tours have at least one client
        attempts = 0
        while (tour1_idx == tour2_idx or 
               not new_solution[tour1_idx] or 
               not new_solution[tour2_idx]):
            tour1_idx = random.randint(0, len(new_solution) - 1)
            tour2_idx = random.randint(0, len(new_solution) - 1)
            attempts += 1
            if attempts > 10:
                return new_solution  # Give up if we can't find valid tours
        
        client1_idx = random.randint(0, len(new_solution[tour1_idx]) - 1)
        client2_idx = random.randint(0, len(new_solution[tour2_idx]) - 1)
        
        client1_id = new_solution[tour1_idx][client1_idx]
        client2_id = new_solution[tour2_idx][client2_idx]
        
        # Check if swapping would violate capacity constraints
        new_tour1 = new_solution[tour1_idx].copy()
        new_tour2 = new_solution[tour2_idx].copy()
        
        new_tour1[client1_idx] = client2_id
        new_tour2[client2_idx] = client1_id
        
        if (is_valid_tour(new_tour1, clients_dict) and 
            is_valid_tour(new_tour2, clients_dict)):
            new_solution[tour1_idx] = new_tour1
            new_solution[tour2_idx] = new_tour2
    
    else:
        # 2-OPT: Reverse a segment within a tour
        if not new_solution:
            return new_solution
            
        tour_idx = random.randint(0, len(new_solution) - 1)
        if len(new_solution[tour_idx]) < 3:
            return new_solution  # Need at least 3 clients for 2-opt
        
        # Select two positions to reverse the segment between them
        i = random.randint(0, len(new_solution[tour_idx]) - 3)
        j = random.randint(i + 1, len(new_solution[tour_idx]) - 1)
        
        # Reverse the segment
        new_solution[tour_idx][i:j+1] = reversed(new_solution[tour_idx][i:j+1])
    
    return new_solution

def format_solution(tours):
    """Convert tours to the required string format."""
    tours_string = ""
    for tour in tours:
        tours_string += " ".join(map(str, tour)) + "\n"
    return tours_string

def simulated_annealing(clients, initial_temp=5000.0, cooling_rate=0.9995, min_temp=0.001, max_iterations=500000):
    """Solve the VRP using simulated annealing."""
    # Create an initial solution
    current_solution, clients_dict = create_initial_solution(clients)
    best_solution = deepcopy(current_solution)
    
    current_distance = calculate_total_distance(current_solution, clients_dict)
    best_distance = current_distance
    
    temperature = initial_temp
    iteration = 0
    stagnation_count = 0
    last_improvement = 0
    
    # Track all-time best
    all_time_best_distance = float('inf')
    all_time_best_solution = None
    all_time_best_string = ""
    
    print(f"Initial solution distance: {current_distance}")
    
    # Different neighborhood types
    neighborhood_types = 3
    current_type = 0
    
    while temperature > min_temp and iteration < max_iterations:
        iteration += 1
        
        # Every 5000 iterations, validate the current best solution using the evaluator
        if iteration % 5000 == 0:
            solution_string = format_solution(best_solution)
            score, valid, message = get_score(solution_string)
            
            print(f"Iteration {iteration}, Temperature: {temperature:.2f}")
            print(f"Current best distance: {best_distance}, Evaluator score: {score}")
            
            if valid and score < all_time_best_distance:
                all_time_best_distance = score
                all_time_best_solution = deepcopy(best_solution)
                all_time_best_string = solution_string
                
                # Save the best solution to file
                date = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_name = f'sol_{score}_solve_v7_{date}'
                
                print(f"New all-time best score: {score}, saving to {file_name}")
                
                with open(f'solutions/{file_name}.txt', 'w') as f:
                    f.write(solution_string)
        
        # Generate a neighbor solution
        neighbor_solution = get_neighbor(current_solution, clients_dict, current_type)
        neighbor_distance = calculate_total_distance(neighbor_solution, clients_dict)
        
        # Decide whether to accept the neighbor solution
        if neighbor_distance < current_distance:
            # Accept better solution
            current_solution = neighbor_solution
            current_distance = neighbor_distance
            
            # Update best solution if needed
            if current_distance < best_distance:
                best_solution = deepcopy(current_solution)
                best_distance = current_distance
                last_improvement = iteration
                stagnation_count = 0
                
                if iteration % 1000 == 0:
                    print(f"New best distance: {best_distance}")
            else:
                stagnation_count += 1
        else:
            # Accept worse solution with a probability that decreases with temperature
            delta = neighbor_distance - current_distance
            acceptance_probability = math.exp(-delta / temperature)
            
            if random.random() < acceptance_probability:
                current_solution = neighbor_solution
                current_distance = neighbor_distance
                stagnation_count += 1
            else:
                stagnation_count += 1
        
        # Cooling schedule
        temperature *= cooling_rate
        
        # Change neighborhood type periodically
        if iteration % 1000 == 0:
            current_type = (current_type + 1) % neighborhood_types
        
        # If no improvement for a while, restart from the best solution with higher temperature
        if stagnation_count > 5000:
            print(f"Restarting at iteration {iteration} due to stagnation")
            current_solution = deepcopy(best_solution)
            current_distance = best_distance
            temperature = initial_temp * 0.7
            stagnation_count = 0
    
    print(f"Simulated annealing completed after {iteration} iterations")
    print(f"Best internal distance: {best_distance}")
    print(f"Best validated score: {all_time_best_distance}")
    
    # Return the best solution found during the search
    return all_time_best_string if all_time_best_string else format_solution(best_solution)

def solve_v7(clients: list[dict[str, any]]) -> str:
    """Solve the pizza delivery problem using simulated annealing."""
    return simulated_annealing(clients)