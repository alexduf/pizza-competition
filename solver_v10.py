import random
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from copy import deepcopy
from functools import partial

from evaluator import manhattan_distance, get_score, depot, capacity

# Hybrid approach combining solver_v5's strategy with genetic algorithm

def next_delivery_candidates(
        position: tuple[int, int],
        clients: dict,
        client_ids: list[int],
        available_space: int
) -> list[int]:
    """Find next clients using solver_v5's nearest neighbor approach."""
    # Sort clients by distance
    sorted_ids = sorted(client_ids, 
                         key=lambda cid: manhattan_distance(position, clients[cid]["position"]))
    
    # Filter to keep only clients we have capacity for
    candidates = [cid for cid in sorted_ids if clients[cid]["pizzas"] <= available_space]
    
    # Return the best candidate, or empty list if none found
    return candidates[:1]

def first_delivery_candidates(
        position: tuple[int, int],
        clients: dict,
        client_ids: list[int],
        available_space: int
) -> list[int]:
    """Find first clients using solver_v5's farthest-first approach."""
    # Sort clients by distance (farthest first)
    sorted_ids = sorted(client_ids, 
                        key=lambda cid: manhattan_distance(position, clients[cid]["position"]),
                        reverse=True)
    
    # Return the top 10 candidates (no capacity filter for first client as in solver_v5)
    return sorted_ids[:15]

def plan_tour_v5(
        clients: dict,
        client_ids: list[int]
) -> list[int]:
    """Plan a single tour using solver_v5's approach."""
    available_space = capacity
    current_position = depot
    current_tour = []
    remaining_ids = client_ids.copy()
    
    while remaining_ids:
        if available_space <= 0:
            return current_tour
        
        if len(current_tour) == 0:
            # First client: pick from farthest candidates
            candidates = first_delivery_candidates(current_position, clients, remaining_ids, available_space)
            if not candidates:
                return current_tour
            
            # Pick one random candidate from the farthest ones
            selected_idx = random.randint(0, min(len(candidates) - 1, 9))
            neighbor_id = candidates[selected_idx]
        else:
            # Subsequent clients: pick nearest
            candidates = next_delivery_candidates(current_position, clients, remaining_ids, available_space)
            if not candidates:
                return current_tour
            
            neighbor_id = candidates[0]  # Take the nearest one
        
        # Early termination condition from solver_v5
        neighbor_pos = clients[neighbor_id]["position"]
        if manhattan_distance(current_position, neighbor_pos) > manhattan_distance(depot, neighbor_pos):
            return current_tour
        
        # Add client to tour
        current_position = neighbor_pos
        available_space -= clients[neighbor_id]["pizzas"]
        remaining_ids.remove(neighbor_id)
        current_tour.append(neighbor_id)
    
    return current_tour

def build_solution_v5(clients_dict):
    """Build a complete solution using solver_v5's approach."""
    all_client_ids = list(clients_dict.keys())
    tours = []
    
    while all_client_ids:
        tour = plan_tour_v5(clients_dict, all_client_ids)
        if not tour:
            break
            
        tours.append(tour)
        for client_id in tour:
            all_client_ids.remove(client_id)
    
    return tours

def evaluate_fitness(solution_with_clients_dict):
    """Calculate fitness (total distance) for a solution."""
    solution, clients_dict = solution_with_clients_dict
    total_distance = 0
    
    for tour in solution:
        # Start from depot
        current_pos = depot
        
        # Visit all clients in the tour
        for client_id in tour:
            client_pos = clients_dict[client_id]["position"]
            total_distance += manhattan_distance(current_pos, client_pos)
            current_pos = client_pos
            
        # Return to depot
        total_distance += manhattan_distance(current_pos, depot)
        
    return total_distance

class HybridGeneticAlgorithm:
    def __init__(self, clients, population_size=200, crossover_rate=0.8, mutation_rate=0.3):
        self.clients = clients
        self.clients_dict = {client["id"]: client for client in clients}
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.best_solution = None
        self.best_fitness = float('inf')
        self.all_time_best_solution = None
        self.all_time_best_fitness = float('inf')
        self.generation = 0
        self.num_processors = max(1, mp.cpu_count() - 1)  # Leave one core free
        print(f"Using {self.num_processors} CPU cores for parallel processing")
        
    def initialize_population(self):
        """Generate initial population with diverse solutions."""
        population = []
        
        # Generate 50% of population using solver_v5 approach with randomness
        for _ in range(self.population_size // 2):
            solution = build_solution_v5(self.clients_dict)
            population.append(solution)
            
        # Generate other 50% using random approach (more diverse)
        for _ in range(self.population_size - len(population)):
            solution = self._generate_random_solution()
            population.append(solution)
            
        return population
    
    def _generate_random_solution(self):
        """Generate a random feasible solution."""
        unassigned = list(self.clients_dict.keys())
        tours = []
        
        while unassigned:
            # Start a new tour
            tour = []
            remaining_capacity = capacity
            current_pos = depot
            
            # Choose first client using solver_v5 approach
            if unassigned:
                first_candidates = first_delivery_candidates(current_pos, self.clients_dict, unassigned, remaining_capacity)
                if first_candidates:
                    selected_idx = random.randint(0, min(len(first_candidates) - 1, 9))
                    client_id = first_candidates[selected_idx]
                    unassigned.remove(client_id)
                    tour.append(client_id)
                    remaining_capacity -= self.clients_dict[client_id]["pizzas"]
                    current_pos = self.clients_dict[client_id]["position"]
            
            # Add more clients to tour until capacity is reached
            added = True
            while added and unassigned:
                added = False
                next_candidates = sorted(unassigned, 
                                        key=lambda cid: manhattan_distance(current_pos, self.clients_dict[cid]["position"]))
                
                for client_id in next_candidates:
                    if self.clients_dict[client_id]["pizzas"] <= remaining_capacity:
                        client_pos = self.clients_dict[client_id]["position"]
                        
                        # Apply early termination check
                        if manhattan_distance(current_pos, client_pos) > manhattan_distance(depot, client_pos):
                            continue
                            
                        tour.append(client_id)
                        unassigned.remove(client_id)
                        remaining_capacity -= self.clients_dict[client_id]["pizzas"]
                        current_pos = client_pos
                        added = True
                        break
            
            if tour:  # Add non-empty tour to solution
                tours.append(tour)
                
        return tours
    
    def evaluate_population_parallel(self, population):
        """Evaluate fitness of all solutions in the population in parallel."""
        with ProcessPoolExecutor(max_workers=self.num_processors) as executor:
            fitnesses = list(executor.map(
                evaluate_fitness, 
                [(solution, self.clients_dict) for solution in population]
            ))
            
        return fitnesses
    
    def select_parents(self, population, fitnesses):
        """Select parents using tournament selection."""
        tournament_size = 3
        parents = []
        
        for _ in range(2):  # Select two parents
            # Random tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            
            # Select winner (lowest fitness/distance)
            winner_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
            parents.append(deepcopy(population[winner_idx]))
            
        return parents
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create offspring."""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
            
        # Use route-based crossover
        # 1. Choose random points to exchange routes
        child1 = []
        child2 = []
        
        # Choose random split points
        split1 = random.randint(0, len(parent1))
        split2 = random.randint(0, len(parent2))
        
        # Create children by crossing routes
        child1 = deepcopy(parent1[:split1])
        child2 = deepcopy(parent2[:split2])
        
        # Find clients already in child1
        clients_in_child1 = set()
        for tour in child1:
            for client_id in tour:
                clients_in_child1.add(client_id)
                
        # Find clients already in child2
        clients_in_child2 = set()
        for tour in child2:
            for client_id in tour:
                clients_in_child2.add(client_id)
                
        # Add missing clients from parent2 to child1
        for tour in parent2:
            new_tour = []
            for client_id in tour:
                if client_id not in clients_in_child1:
                    new_tour.append(client_id)
                    clients_in_child1.add(client_id)
                    
            if new_tour:
                child1.append(new_tour)
                
        # Add missing clients from parent1 to child2
        for tour in parent1:
            new_tour = []
            for client_id in tour:
                if client_id not in clients_in_child2:
                    new_tour.append(client_id)
                    clients_in_child2.add(client_id)
                    
            if new_tour:
                child2.append(new_tour)
                
        # Verify all clients are included in both children
        self._verify_solution(child1)
        self._verify_solution(child2)
                
        return child1, child2
    
    def _verify_solution(self, solution):
        """Verify all clients are included in solution and fix if needed."""
        client_ids = set(self.clients_dict.keys())
        
        # Find all clients in solution
        included_clients = set()
        for tour in solution:
            for client_id in tour:
                included_clients.add(client_id)
                
        # Find missing clients
        missing_clients = client_ids - included_clients
        
        # Add missing clients to solution
        if missing_clients:
            # Collect client info for missing clients
            missing_client_objects = []
            for client_id in missing_clients:
                missing_client_objects.append({
                    "id": client_id,
                    "pizzas": self.clients_dict[client_id]["pizzas"],
                    "position": self.clients_dict[client_id]["position"]
                })
                
            # Sort by pizza count (smaller first to fit easier)
            missing_client_objects.sort(key=lambda c: c["pizzas"])
            
            # Try to fit each missing client into existing tours
            for client in missing_client_objects:
                client_id = client["id"]
                client_pizzas = client["pizzas"]
                
                # Try to find a tour with enough capacity
                placed = False
                for tour in solution:
                    tour_pizzas = sum(self.clients_dict[cid]["pizzas"] for cid in tour)
                    if tour_pizzas + client_pizzas <= capacity:
                        # Use v5's approach: if the missing client is closer to the depot
                        # than the last client in the tour, add it at the end
                        if tour:
                            last_client_id = tour[-1]
                            last_client_pos = self.clients_dict[last_client_id]["position"]
                            client_pos = self.clients_dict[client_id]["position"]
                            
                            # Only add if it's not better to return to depot first
                            if manhattan_distance(last_client_pos, client_pos) <= manhattan_distance(depot, client_pos):
                                tour.append(client_id)
                                placed = True
                                break
                        else:
                            # Empty tour, just add the client
                            tour.append(client_id)
                            placed = True
                            break
                        
                # If not placed, create a new tour
                if not placed:
                    solution.append([client_id])
                    
        # Check for duplicate clients
        all_clients = []
        for tour in solution:
            all_clients.extend(tour)
            
        if len(all_clients) != len(set(all_clients)):
            # There are duplicates - fix by removing duplicates
            seen = set()
            for tour in solution:
                i = 0
                while i < len(tour):
                    if tour[i] in seen:
                        tour.pop(i)
                    else:
                        seen.add(tour[i])
                        i += 1
                        
        # Remove empty tours
        i = 0
        while i < len(solution):
            if not solution[i]:
                solution.pop(i)
            else:
                i += 1
                
        # Optimize the order within each tour
        for tour in solution:
            self._optimize_tour_order(tour)
    
    def _optimize_tour_order(self, tour):
        """Optimize the order of clients within a tour using nearest neighbor."""
        if len(tour) <= 1:
            return
        
        # Try keeping the first client fixed (solver_v5 approach)
        first_client_id = tour[0]
        remaining = tour[1:]
        optimized = [first_client_id]
        current_pos = self.clients_dict[first_client_id]["position"]
        
        while remaining:
            # Find nearest unvisited client
            nearest_idx = 0
            min_dist = float('inf')
            
            for i, client_id in enumerate(remaining):
                client_pos = self.clients_dict[client_id]["position"]
                dist = manhattan_distance(current_pos, client_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
                    
            # Early termination check
            next_client_id = remaining[nearest_idx]
            next_client_pos = self.clients_dict[next_client_id]["position"]
            if manhattan_distance(current_pos, next_client_pos) > manhattan_distance(depot, next_client_pos):
                break
                
            # Add nearest client to optimized tour
            client_id = remaining.pop(nearest_idx)
            optimized.append(client_id)
            current_pos = self.clients_dict[client_id]["position"]
        
        # Add any remaining clients
        for client_id in remaining:
            optimized.append(client_id)
            
        # Replace original tour with optimized one
        tour[:] = optimized
    
    def mutate(self, solution):
        """Apply mutation operators to a solution."""
        if random.random() > self.mutation_rate:
            return solution
            
        # Choose a random mutation operator
        mutation_type = random.choice([
            self._swap_within_tour,
            self._swap_between_tours,
            self._relocate_client,
            self._reoptimize_tour,   # New mutation that applies v5's logic
            self._recreate_tour      # New mutation that recreates a tour using v5's approach
        ])
        
        # Apply mutation
        mutation_type(solution)
        
        # Ensure solution is valid after mutation
        self._verify_solution(solution)
        
        return solution
    
    def _swap_within_tour(self, solution):
        """Swap two clients within the same tour."""
        if not solution:
            return
            
        # Select a non-empty tour with at least 2 clients
        non_empty_tours = [i for i, tour in enumerate(solution) if len(tour) >= 2]
        if not non_empty_tours:
            return
            
        tour_idx = random.choice(non_empty_tours)
        tour = solution[tour_idx]
        
        # Don't swap the first client (keep solver_v5's farthest-first strategy)
        if len(tour) == 2:
            return  # Can't swap without affecting first client
            
        # Select two positions to swap, preserving first client
        pos1 = random.randint(1, len(tour) - 1)
        pos2 = random.randint(1, len(tour) - 1)
        while pos1 == pos2:
            pos2 = random.randint(1, len(tour) - 1)
            
        # Swap clients
        tour[pos1], tour[pos2] = tour[pos2], tour[pos1]
    
    def _swap_between_tours(self, solution):
        """Swap clients between two different tours."""
        if len(solution) < 2:
            return
            
        # Select two different tours
        tour1_idx = random.randint(0, len(solution) - 1)
        tour2_idx = random.randint(0, len(solution) - 1)
        while tour1_idx == tour2_idx:
            tour2_idx = random.randint(0, len(solution) - 1)
            
        tour1 = solution[tour1_idx]
        tour2 = solution[tour2_idx]
        
        if not tour1 or not tour2:
            return
            
        # Try not to swap first clients (solver_v5 strategy)
        client1_idx = random.randint(1 if len(tour1) > 1 else 0, len(tour1) - 1)
        client2_idx = random.randint(1 if len(tour2) > 1 else 0, len(tour2) - 1)
        
        client1_id = tour1[client1_idx]
        client2_id = tour2[client2_idx]
        
        # Check if the swap violates capacity constraints
        pizzas_tour1 = sum(self.clients_dict[cid]["pizzas"] for cid in tour1)
        pizzas_tour2 = sum(self.clients_dict[cid]["pizzas"] for cid in tour2)
        
        pizzas_client1 = self.clients_dict[client1_id]["pizzas"]
        pizzas_client2 = self.clients_dict[client2_id]["pizzas"]
        
        new_pizzas_tour1 = pizzas_tour1 - pizzas_client1 + pizzas_client2
        new_pizzas_tour2 = pizzas_tour2 - pizzas_client2 + pizzas_client1
        
        # Only perform swap if capacity constraints are satisfied
        if new_pizzas_tour1 <= capacity and new_pizzas_tour2 <= capacity:
            tour1[client1_idx] = client2_id
            tour2[client2_idx] = client1_id
    
    def _relocate_client(self, solution):
        """Move a client from one tour to another."""
        if len(solution) < 2:
            return
            
        # Select source tour
        from_tour_idx = random.randint(0, len(solution) - 1)
        from_tour = solution[from_tour_idx]
        
        if not from_tour or len(from_tour) <= 1:  # Don't empty tours or move first client
            return
            
        # Select a client to relocate (not the first one unless it's the only one)
        client_idx = random.randint(1, len(from_tour) - 1)
        client_id = from_tour[client_idx]
        client_pizzas = self.clients_dict[client_id]["pizzas"]
        
        # Select destination tour
        to_tour_idx = random.randint(0, len(solution) - 1)
        while to_tour_idx == from_tour_idx:
            to_tour_idx = random.randint(0, len(solution) - 1)
            
        to_tour = solution[to_tour_idx]
        
        # Check if relocation violates capacity constraints
        to_tour_pizzas = sum(self.clients_dict[cid]["pizzas"] for cid in to_tour)
        if to_tour_pizzas + client_pizzas <= capacity:
            # Remove client from source tour
            from_tour.pop(client_idx)
            
            # Add client to destination tour
            to_tour.append(client_id)
            
            # Remove empty tours
            if not from_tour:
                solution.pop(from_tour_idx)
    
    def _reoptimize_tour(self, solution):
        """Re-optimize a random tour using solver_v5's approach."""
        if not solution:
            return
            
        # Select a tour to optimize
        tour_idx = random.randint(0, len(solution) - 1)
        tour = solution[tour_idx]
        
        if not tour:
            return
            
        # Save first client (keep solver_v5's farthest-first strategy)
        first_client_id = tour[0] if tour else None
        
        # Get all clients in the tour
        client_ids = tour.copy()
        
        # Clear the tour
        tour.clear()
        
        # If there was a first client, add it back
        if first_client_id:
            tour.append(first_client_id)
            client_ids.remove(first_client_id)
            
        # Optimize the rest using solver_v5's approach
        if tour:  # If there's at least one client
            current_pos = self.clients_dict[tour[-1]]["position"]
            available_space = capacity - self.clients_dict[tour[-1]]["pizzas"]
            
            while client_ids and available_space > 0:
                # Find nearest client that fits
                candidates = sorted(client_ids, 
                                   key=lambda cid: manhattan_distance(current_pos, self.clients_dict[cid]["position"]))
                
                # Find first client that fits
                next_client_id = None
                for cid in candidates:
                    if self.clients_dict[cid]["pizzas"] <= available_space:
                        next_client_id = cid
                        break
                        
                if not next_client_id:
                    break
                    
                next_pos = self.clients_dict[next_client_id]["position"]
                
                # Early termination check
                if manhattan_distance(current_pos, next_pos) > manhattan_distance(depot, next_pos):
                    break
                    
                # Add client to tour
                tour.append(next_client_id)
                client_ids.remove(next_client_id)
                current_pos = next_pos
                available_space -= self.clients_dict[next_client_id]["pizzas"]
        
        # If there are still clients left, add them to a new tour
        if client_ids:
            # Create a new tour with remaining clients
            new_tour = []
            solution.append(new_tour)
            
            # Find the farthest client to start with
            if client_ids:
                farthest_clients = sorted(client_ids, 
                                         key=lambda cid: manhattan_distance(depot, self.clients_dict[cid]["position"]),
                                         reverse=True)
                
                first_id = farthest_clients[0]
                new_tour.append(first_id)
                client_ids.remove(first_id)
                
                # Optimize rest of tour
                if client_ids:
                    current_pos = self.clients_dict[first_id]["position"]
                    available_space = capacity - self.clients_dict[first_id]["pizzas"]
                    
                    while client_ids and available_space > 0:
                        # Find nearest client
                        candidates = sorted(client_ids, 
                                           key=lambda cid: manhattan_distance(current_pos, self.clients_dict[cid]["position"]))
                        
                        # Find first client that fits
                        next_client_id = None
                        for cid in candidates:
                            if self.clients_dict[cid]["pizzas"] <= available_space:
                                next_client_id = cid
                                break
                                
                        if not next_client_id:
                            break
                            
                        next_pos = self.clients_dict[next_client_id]["position"]
                        
                        # Early termination check
                        if manhattan_distance(current_pos, next_pos) > manhattan_distance(depot, next_pos):
                            break
                            
                        # Add client to tour
                        new_tour.append(next_client_id)
                        client_ids.remove(next_client_id)
                        current_pos = next_pos
                        available_space -= self.clients_dict[next_client_id]["pizzas"]
    
    def _recreate_tour(self, solution):
        """Recreate a tour from scratch using solver_v5's approach."""
        if len(solution) < 2:  # Need at least 2 tours to make this interesting
            return
            
        # Select two random tours to merge and recreate
        tour1_idx = random.randint(0, len(solution) - 1)
        tour2_idx = random.randint(0, len(solution) - 1)
        while tour1_idx == tour2_idx:
            tour2_idx = random.randint(0, len(solution) - 1)
            
        # Get all clients from both tours
        all_clients = []
        for client_id in solution[tour1_idx]:
            all_clients.append(client_id)
        for client_id in solution[tour2_idx]:
            all_clients.append(client_id)
            
        # Remove the original tours (in reverse order to avoid index issues)
        if tour1_idx > tour2_idx:
            solution.pop(tour1_idx)
            solution.pop(tour2_idx)
        else:
            solution.pop(tour2_idx)
            solution.pop(tour1_idx)
            
        # Create new tours using solver_v5's approach
        remaining_clients = all_clients.copy()
        
        while remaining_clients:
            # Plan a tour
            current_tour = []
            available_space = capacity
            current_position = depot
            
            # First client selection (farthest)
            if remaining_clients:
                farthest_clients = sorted(
                    remaining_clients,
                    key=lambda cid: manhattan_distance(current_position, self.clients_dict[cid]["position"]),
                    reverse=True
                )[:10]
                
                if farthest_clients:
                    first_id = random.choice(farthest_clients)
                    current_tour.append(first_id)
                    remaining_clients.remove(first_id)
                    current_position = self.clients_dict[first_id]["position"]
                    available_space -= self.clients_dict[first_id]["pizzas"]
            
            # Add more clients using nearest neighbor
            added = True
            while added and remaining_clients and available_space > 0:
                added = False
                
                # Find nearest client that fits
                nearest_clients = sorted(
                    remaining_clients,
                    key=lambda cid: manhattan_distance(current_position, self.clients_dict[cid]["position"])
                )
                
                for client_id in nearest_clients:
                    if self.clients_dict[client_id]["pizzas"] <= available_space:
                        client_pos = self.clients_dict[client_id]["position"]
                        
                        # Early termination check
                        if manhattan_distance(current_position, client_pos) > manhattan_distance(depot, client_pos):
                            continue
                            
                        current_tour.append(client_id)
                        remaining_clients.remove(client_id)
                        current_position = client_pos
                        available_space -= self.clients_dict[client_id]["pizzas"]
                        added = True
                        break
            
            # Add the tour if not empty
            if current_tour:
                solution.append(current_tour)
    
    def format_solution(self, solution):
        """Convert solution to the required string format."""
        tours_string = ""
        for tour in solution:
            tours_string += " ".join(map(str, tour)) + "\n"
        return tours_string
    
    def save_solution(self, solution_string, score):
        """Save solution to file."""
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'sol_{score}_solve_v10_{date}'
        
        with open(f'solutions/{file_name}.txt', 'w') as f:
            f.write(solution_string)
            
        print(f"Solution saved to {file_name}.txt")
    
    def create_offspring_batch(self, population, fitnesses, batch_size):
        """Create a batch of offspring."""
        offspring = []
        
        for _ in range(batch_size // 2):  # Each iteration creates 2 offspring
            # Select parents
            parent1, parent2 = self.select_parents(population, fitnesses)
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add to offspring
            offspring.append(child1)
            offspring.append(child2)
            
        return offspring
    
    def evolve(self, max_generations=5000, stagnation_limit=300):
        """Run the genetic algorithm with parallel processing."""
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        fitnesses = self.evaluate_population_parallel(population)
        best_idx = fitnesses.index(min(fitnesses))
        self.best_solution = deepcopy(population[best_idx])
        self.best_fitness = fitnesses[best_idx]
        
        # Initialize all-time best
        self.all_time_best_solution = deepcopy(self.best_solution)
        self.all_time_best_fitness = self.best_fitness
        
        print(f"Initial best fitness: {self.best_fitness}")
        
        # Validate initial best solution
        solution_string = self.format_solution(self.best_solution)
        score, valid, message = get_score(solution_string)
        
        if valid:
            print(f"Initial solution validated score: {score}")
            self.all_time_best_solution = deepcopy(self.best_solution)
            self.all_time_best_fitness = score
            
            # Save initial best solution
            self.save_solution(solution_string, score)
        else:
            print(f"Initial solution validation failed: {message}")
            
        stagnation_counter = 0
        
        # Main evolution loop
        for generation in range(max_generations):
            self.generation = generation
            
            # Check for stagnation
            if stagnation_counter >= stagnation_limit:
                print(f"Stagnation detected at generation {generation}, resetting population")
                
                # Keep best solution
                new_population = [deepcopy(self.best_solution)]
                
                # Add some solver_v5 solutions
                for _ in range((self.population_size - 1) // 2):
                    new_population.append(build_solution_v5(self.clients_dict))
                    
                # Add some random solutions
                for _ in range(self.population_size - len(new_population)):
                    new_population.append(self._generate_random_solution())
                    
                population = new_population
                fitnesses = self.evaluate_population_parallel(population)
                stagnation_counter = 0
            
            # Create new population
            new_population = []
            
            # Elitism: always keep the best solution
            new_population.append(deepcopy(self.best_solution))
            
            # Create offspring in parallel batches
            batch_count = self.num_processors
            batch_size = (self.population_size - 1) // batch_count
            
            offspring_batches = []
            batch_range = []
            for i in range(batch_count):
                start = i * batch_size
                end = start + batch_size if i < batch_count - 1 else self.population_size - 1
                batch_range.append((start, end))
                
            with ProcessPoolExecutor(max_workers=self.num_processors) as executor:
                # Create a partial function with population and fitnesses already bound
                create_batch_partial = partial(
                    self.create_offspring_batch, 
                    population, 
                    fitnesses
                )
                
                # Process each batch in parallel
                batch_sizes = [end - start for start, end in batch_range]
                offspring_batches = list(executor.map(create_batch_partial, batch_sizes))
            
            # Combine all offspring batches
            for batch in offspring_batches:
                new_population.extend(batch)
                
            # Ensure population size remains constant
            new_population = new_population[:self.population_size]
            
            # Update population
            population = new_population
            
            # Evaluate new population in parallel
            fitnesses = self.evaluate_population_parallel(population)
            current_best_idx = fitnesses.index(min(fitnesses))
            current_best_fitness = fitnesses[current_best_idx]
            
            # Update best solution if improved
            if current_best_fitness < self.best_fitness:
                self.best_solution = deepcopy(population[current_best_idx])
                self.best_fitness = current_best_fitness
                print(f"Generation {generation}: New best fitness: {self.best_fitness}")
                stagnation_counter = 0
                
                # Validate and potentially save new best solution
                if generation % 50 == 0 or current_best_fitness < self.all_time_best_fitness * 0.95:
                    solution_string = self.format_solution(self.best_solution)
                    score, valid, message = get_score(solution_string)
                    
                    if valid and score < self.all_time_best_fitness:
                        print(f"Generation {generation}: New validated best score: {score}")
                        self.all_time_best_solution = deepcopy(self.best_solution)
                        self.all_time_best_fitness = score
                        
                        # Save new best solution
                        self.save_solution(solution_string, score)
            else:
                stagnation_counter += 1
                
            # Periodically try solver_v5 approach to inject good solutions
            if generation % 100 == 0:
                for _ in range(5):  # Try 5 random v5 solutions
                    v5_solution = build_solution_v5(self.clients_dict)
                    v5_fitness = evaluate_fitness((v5_solution, self.clients_dict))
                    
                    if v5_fitness < self.best_fitness:
                        print(f"Generation {generation}: Found better v5 solution: {v5_fitness}")
                        self.best_solution = deepcopy(v5_solution)
                        self.best_fitness = v5_fitness
                        stagnation_counter = 0
                        
                        # Validate and potentially save
                        solution_string = self.format_solution(v5_solution)
                        score, valid, message = get_score(solution_string)
                        
                        if valid and score < self.all_time_best_fitness:
                            print(f"Generation {generation}: New validated v5 score: {score}")
                            self.all_time_best_solution = deepcopy(v5_solution)
                            self.all_time_best_fitness = score
                            
                            # Save new best solution
                            self.save_solution(solution_string, score)
            
            # Log progress
            if generation % 100 == 0:
                print(f"Generation {generation}: Best fitness: {self.best_fitness}, All-time best: {self.all_time_best_fitness}")
        
        # Final evaluation
        solution_string = self.format_solution(self.best_solution)
        score, valid, message = get_score(solution_string)
        
        print("Genetic algorithm completed")
        print(f"Best internal fitness: {self.best_fitness}")
        print(f"Best validated score: {self.all_time_best_fitness}")
        
        # Return the best validated solution
        if self.all_time_best_solution:
            return self.format_solution(self.all_time_best_solution)
        return solution_string

def solve_v10(clients: list[dict[str, any]]) -> str:
    """Solve the pizza delivery problem using a hybrid approach combining solver_v5 with genetic algorithm."""
    hga = HybridGeneticAlgorithm(
        clients=clients,
        population_size=200,
        crossover_rate=0.8,
        mutation_rate=0.3
    )
    return hga.evolve(max_generations=5000, stagnation_limit=300)