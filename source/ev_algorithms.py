import numpy as np
import source.helpers as hp


def differential_evolution(
    objective_function,
    bounds,
    population_size=50,
    max_generations=100,
    F=0.5,
    CR=0.7,
    strategy="best/1/bin",
):
    num_dimensions = len(bounds)
    population = np.random.uniform(
        bounds[:, 0], bounds[:, 1], size=(population_size, num_dimensions)
    )
    best_solution = population[
        np.argmin([objective_function(candidate) for candidate in population])
    ]

    for generation in range(max_generations):
        new_population = []

        for i in range(population_size):
            target = population[i]

            if strategy == "best/1/bin":
                a, b = np.random.choice(population, 2, replace=False)
                best = best_solution
                mutant = best + F * (a - b)
            elif strategy == "rand/1/bin":
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant = a + F * (b - c)
            else:
                raise ValueError("Invalid strategy")

            crossover_mask = np.random.rand(num_dimensions) < CR
            trial = np.where(crossover_mask, mutant, target)

            if objective_function(trial) < objective_function(target):
                new_population.append(trial)
            else:
                new_population.append(target)

        population = np.array(new_population)
        best_solution = population[
            np.argmin([objective_function(candidate) for candidate in population])
        ]

    return best_solution, objective_function(best_solution)


def particle_swarm_optimization(
    objective_function,
    num_particles,
    num_dimensions,
    max_iterations,
    inertia_weight=0.7298,
    cognitive_weight=1.49618,
    social_weight=1.49618,
):
    particles = [hp.Particle(num_dimensions) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float("inf")

    for iteration in range(max_iterations):
        for particle in particles:
            value = objective_function(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position

            if value < global_best_value:
                global_best_value = value
                global_best_position = particle.position

        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()
            particle.velocity = (
                inertia_weight * particle.velocity
                + cognitive_weight * r1 * (particle.best_position - particle.position)
                + social_weight * r2 * (global_best_position - particle.position)
            )
            particle.position += particle.velocity

    return global_best_position, global_best_value


def soma_all_to_one(
    objective_function,
    bounds,
    num_populations=5,
    population_size=50,
    max_generations=100,
    F=0.5,
    CR=0.7,
    num_particles=30,
    inertia_weight=0.7,
    cognitive_weight=1.5,
    social_weight=1.5,
):
    best_global_solution = None
    best_global_value = float("inf")

    for _ in range(num_populations):
        best_solution, best_value = differential_evolution(
            objective_function, bounds, population_size, max_generations, F, CR
        )
        if best_value < best_global_value:
            best_global_value = best_value
            best_global_solution = best_solution

    global_best_position = best_global_solution
    num_dimensions = len(bounds)
    particles = [hp.Particle(num_dimensions) for _ in range(num_particles)]

    for iteration in range(max_generations):
        for particle in particles:
            value = objective_function(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position

            if value < best_global_value:
                best_global_value = value
                global_best_position = particle.position

        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()
            particle.velocity = (
                inertia_weight * particle.velocity
                + cognitive_weight * r1 * (particle.best_position - particle.position)
                + social_weight * r2 * (global_best_position - particle.position)
            )
            particle.position += particle.velocity

    return best_global_solution, best_global_value


def soma_all_to_all(
    objective_function,
    bounds,
    num_populations=5,
    population_size=50,
    max_generations=100,
    F=0.5,
    CR=0.7,
    num_particles=30,
    inertia_weight=0.7,
    cognitive_weight=1.5,
    social_weight=1.5,
):
    best_global_solution = None
    best_global_value = float("inf")

    populations = []

    for _ in range(num_populations):
        best_solution, best_value = differential_evolution(
            objective_function, bounds, population_size, max_generations, F, CR
        )
        if best_value < best_global_value:
            best_global_value = best_value
            best_global_solution = best_solution
        populations.append((best_solution, best_value))

    global_best_position = best_global_solution
    num_dimensions = len(bounds)
    particles = [hp.Particle(num_dimensions) for _ in range(num_particles)]

    for iteration in range(max_generations):
        for i, particle in enumerate(particles):
            value = objective_function(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position

            # Update global best based on best of all populations
            if value < best_global_value:
                best_global_value = value
                global_best_position = particle.position

            # Update local best for this population based on the best in its population
            if value < populations[i][1]:
                populations[i] = (particle.position, value)

        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()
            particle.velocity = (
                inertia_weight * particle.velocity
                + cognitive_weight * r1 * (particle.best_position - particle.position)
                + social_weight * r2 * (global_best_position - particle.position)
            )
            particle.position += particle.velocity

    return best_global_solution, best_global_value
