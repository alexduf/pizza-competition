import random
import math
from datetime import datetime
from copy import deepcopy

from evaluator import manhattan_distance, get_score, depot, capacity

# Genetic Algorithm for Vehicle Routing Problem

class GeneticAlgorithm:
    def __init__(self, clients, population_size=50, crossover_rate=0.8, mutation_rate=0.3):
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
        
    def initialize_population(self):
        """Generate initial population with diverse solutions."""
        population = []
        
        # Generate diverse solutions for initial population
        for _ in range(self.population_size):
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
            
            # Prioritize distant clients first with some randomness
            shuffle_idx = min(len(unassigned), 10)
            if shuffle_idx > 0:
                shuffled = unassigned[:shuffle_idx]
                random.shuffle(shuffled)
                unassigned[:shuffle_idx] = shuffled
            
            # Keep adding clients to the current tour until capacity is reached
            i = 0
            while i < len(unassigned):
                client_id = unassigned[i]
                client_pizzas = self.clients_dict[client_id]["pizzas"]
                
                # Check if client fits in current tour
                if client_pizzas <= remaining_capacity:
                    tour.append(client_id)
                    remaining_capacity -= client_pizzas
                    unassigned.pop(i)
                else:
                    i += 1
                    
                # Break if we can't add more clients or tour is getting too large
                if i >= len(unassigned) or len(tour) >= 10:
                    break
                    
            if tour:  # Add non-empty tour to solution
                # Optimize order within tour using nearest neighbor
                self._optimize_tour(tour)
                tours.append(tour)
                
        return tours
    
    def _optimize_tour(self, tour):
        """Optimize the order of clients within a tour using nearest neighbor."""
        if len(tour) <= 1:
            return
            
        # Start from depot
        optimized = []
        remaining = tour.copy()
        current_pos = depot
        
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
            
            # Add nearest client to optimized tour
            client_id = remaining.pop(nearest_idx)
            optimized.append(client_id)
            current_pos = self.clients_dict[client_id]["position"]
            
        # Replace original tour with optimized one
        tour[:] = optimized
                
    def evaluate_fitness(self, solution):
        """Calculate fitness (total distance) for a solution."""
        total_distance = 0
        
        for tour in solution:
            # Start from depot
            current_pos = depot
            
            # Visit all clients in the tour
            for client_id in tour:
                client_pos = self.clients_dict[client_id]["position"]
                total_distance += manhattan_distance(current_pos, client_pos)
                current_pos = client_pos
                
            # Return to depot
            total_distance += manhattan_distance(current_pos, depot)
            
        return total_distance
    
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
                missing_client_objects.append(self.clients_dict[client_id])
                
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
    
    def mutate(self, solution):
        """Apply mutation operators to a solution."""
        if random.random() > self.mutation_rate:
            return solution
            
        # Choose a random mutation operator
        mutation_type = random.choice([
            self._swap_within_tour,
            self._swap_between_tours,
            self._relocate_client,
            self._two_opt
        ])
        
        # Apply mutation
        mutation_type(solution)
        
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
        
        # Select two positions to swap
        pos1 = random.randint(0, len(tour) - 1)
        pos2 = random.randint(0, len(tour) - 1)
        while pos1 == pos2:
            pos2 = random.randint(0, len(tour) - 1)
            
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
            
        # Select a random client from each tour
        client1_idx = random.randint(0, len(tour1) - 1)
        client2_idx = random.randint(0, len(tour2) - 1)
        
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
        
        if not from_tour:
            return
            
        # Select a client to relocate
        client_idx = random.randint(0, len(from_tour) - 1)
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
    
    def _two_opt(self, solution):
        """Apply 2-opt operation to a random tour."""
        if not solution:
            return
            
        # Select a tour with at least 3 clients
        valid_tours = [i for i, tour in enumerate(solution) if len(tour) >= 3]
        if not valid_tours:
            return
            
        tour_idx = random.choice(valid_tours)
        tour = solution[tour_idx]
        
        # Select two positions to reverse the segment between them
        i = random.randint(0, len(tour) - 3)
        j = random.randint(i + 1, len(tour) - 1)
        
        # Reverse the segment
        tour[i:j+1] = reversed(tour[i:j+1])
    
    def evolve(self, max_generations=1000, stagnation_limit=200):
        """Run the genetic algorithm."""
        population = self.initialize_population()
        
        # Evaluate initial population
        fitnesses = [self.evaluate_fitness(solution) for solution in population]
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
                
                # Keep best solution and generate new random solutions
                new_population = [deepcopy(self.best_solution)]
                for _ in range(self.population_size - 1):
                    new_population.append(self._generate_random_solution())
                    
                population = new_population
                fitnesses = [self.evaluate_fitness(solution) for solution in population]
                stagnation_counter = 0
            
            # Create new population
            new_population = []
            
            # Elitism: always keep the best solution
            new_population.append(deepcopy(self.best_solution))
            
            # Fill the rest with offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitnesses)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Update population
            population = new_population
            
            # Evaluate new population
            fitnesses = [self.evaluate_fitness(solution) for solution in population]
            current_best_idx = fitnesses.index(min(fitnesses))
            current_best_fitness = fitnesses[current_best_idx]
            
            # Update best solution if improved
            if current_best_fitness < self.best_fitness:
                self.best_solution = deepcopy(population[current_best_idx])
                self.best_fitness = current_best_fitness
                print(f"Generation {generation}: New best fitness: {self.best_fitness}")
                stagnation_counter = 0
                
                # Validate and potentially save new best solution
                if generation % 10 == 0 or current_best_fitness < self.all_time_best_fitness * 0.98:
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
                
            # Log progress
            if generation % 50 == 0:
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
    
    def format_solution(self, solution):
        """Convert solution to the required string format."""
        tours_string = ""
        for tour in solution:
            tours_string += " ".join(map(str, tour)) + "\n"
        return tours_string
    
    def save_solution(self, solution_string, score):
        """Save solution to file."""
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'sol_{score}_solve_v8_{date}'
        
        with open(f'solutions/{file_name}.txt', 'w') as f:
            f.write(solution_string)
            
        print(f"Solution saved to {file_name}.txt")

def solve_v8(clients: list[dict[str, any]]) -> str:
    """Solve the pizza delivery problem using a genetic algorithm."""
    ga = GeneticAlgorithm(
        clients=clients,
        population_size=100,
        crossover_rate=0.8,
        mutation_rate=0.3
    )
    return ga.evolve(max_generations=1000, stagnation_limit=200)