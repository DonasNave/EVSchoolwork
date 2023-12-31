import sys
import numpy as np
import source.helpers as hp

MAX_VAL = sys.float_info.max


def differential_evolution(
    objective_function,
    bounds,
    population_size=50,
    max_evaluations=2000,
    F=0.5,
    CR=0.9,
    strategy="best/1/bin",
):
    # Initialize population
    num_dimensions = len(bounds)
    population = np.random.uniform(
        bounds[:, 0], bounds[:, 1], size=(population_size, num_dimensions)
    )
    # Initialize best solution
    best_solution = population[
        np.argmin([objective_function(candidate) for candidate in population])
    ]

    evaluations = 0

    while evaluations < max_evaluations:
        new_population = []

        for i in range(population_size):
            target = population[i]

            # Act out strategy
            if strategy == "best/1/bin":
                indices = np.random.choice(population_size, 2, replace=False)
                a, b = population[indices]
                best = best_solution
                mutant = best + F * (a - b)
            elif strategy == "rand/1/bin":
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = a + F * (b - c)
            else:
                raise ValueError("Invalid strategy")

            # Bounce
            mutant = hp.bounce_vector(mutant, bounds)

            # Crossover
            crossover_mask = np.random.rand(num_dimensions) < CR
            trial = np.where(crossover_mask, mutant, target)

            # Selection
            if objective_function(trial) < objective_function(target):
                new_population.append(trial)
            else:
                new_population.append(target)

        # Update population, best solution
        population = np.array(new_population)
        best_solution = population[
            np.argmin([objective_function(candidate) for candidate in population])
        ]

        # Update evaluations
        evaluations += population_size

    population = []
    return best_solution, objective_function(best_solution)


def particle_swarm_optimization(
    objective_function,
    bounds,
    num_particles,
    max_evaluations,
    inertia_weight=0.7298,
    cognitive_weight=1.49618,
    social_weight=1.49618,
):
    num_dimensions = len(bounds)
    particles = [hp.Particle(num_dimensions, bounds) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = MAX_VAL
    evaluations = 0

    while evaluations < max_evaluations:
        # Update values
        for particle in particles:
            value = objective_function(particle.position)

            # Update local best
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position

            # Update global best
            if value < global_best_value:
                global_best_value = value
                global_best_position = particle.position

        # Update positions
        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()
            particle.velocity = (
                inertia_weight * particle.velocity
                + cognitive_weight * r1 * (particle.best_position - particle.position)
                + social_weight * r2 * (global_best_position - particle.position)
            )
            particle.position = hp.bounce_vector(
                particle.position + particle.velocity, bounds=bounds
            )

        # Update evaluations
        evaluations += num_particles

    return global_best_position, global_best_value


def soma_ato(
    objective_function,
    bounds,
    population_size,
    max_evaluations=2000,
    path_length=3,
    step=0.11,
    prt=0.7,
):
    num_dimensions = len(bounds)
    particles = np.random.uniform(
        bounds[:, 0], bounds[:, 1], size=(population_size, num_dimensions)
    )
    global_best_position = None
    global_best_value = MAX_VAL

    evaluations = 0

    # To better compare with other algorithms, evaluations are counted instead of migrations
    while evaluations < max_evaluations:
        new_particles = []
        # Update values
        for particle in particles:
            value = objective_function(particle)

            # Update global best
            if value < global_best_value:
                global_best_value = value
                global_best_position = particle

        # Migrate particles
        for particle in particles:
            # Skip leader
            if np.all(particle == global_best_position):
                evaluations += 1
                new_particles.append(particle)
                continue

            # Setup best particle position in steps as current one
            best_particle = particle
            best_value = objective_function(particle)

            # Crete migration steps
            for t in np.arange(0, path_length, step):
                # Generate PRT vector
                random_numbers = [np.random.random() for _ in range(num_dimensions)]
                prt_vector = [0 if num > prt else 1 for num in random_numbers]
                # Calculate direction
                particle_direction = (global_best_position - particle) * prt_vector * t

                moved_particle = particle + particle_direction
                moved_particle = hp.bounce_vector(moved_particle, bounds)
                mutant_value = objective_function(moved_particle)

                if mutant_value < best_value:
                    best_value = mutant_value
                    best_particle = moved_particle

                evaluations += 1

            new_particles.append(best_particle)

        # Update particles to migrated positions
        particles = new_particles

    return global_best_position, global_best_value


def soma_ata(
    objective_function,
    bounds,
    population_size,
    max_evaluations=2000,
    path_length=3,
    step=0.11,
    prt=0.7,
):
    num_dimensions = len(bounds)

    particles = np.random.uniform(
        bounds[:, 0], bounds[:, 1], size=(population_size, num_dimensions)
    )

    global_best_position = None
    global_best_value = MAX_VAL

    evaluations = 0

    # To better compare with other algorithms, evaluations are counted instead of migrations
    while evaluations < max_evaluations:
        new_particles = []

        # Migrate all particles
        for particle in particles:
            # Setup best particle position
            best_particle = particle
            best_value = objective_function(particle)

            # For each other particle as leader
            for leader in particles:
                global_best_position = leader

                # Skip leader
                if np.all(particle == leader):
                    evaluations += 1
                    continue

                # Crete migration steps
                for t in np.arange(0, path_length, step):
                    # Generate PRT vector
                    random_numbers = [np.random.random() for _ in range(num_dimensions)]
                    prt_vector = [0 if num > prt else 1 for num in random_numbers]
                    # Calculate direction
                    particle_direction = (
                        (global_best_position - particle) * prt_vector * t
                    )

                    moved_particle = particle + particle_direction
                    moved_particle = hp.bounce_vector(moved_particle, bounds)
                    mutant_value = objective_function(moved_particle)

                    if mutant_value < best_value:
                        best_value = mutant_value
                        best_particle = moved_particle

                    if best_value < global_best_value:
                        global_best_value = best_value
                        global_best_position = best_particle

                    evaluations += 1

            new_particles.append(best_particle)

        # Update particles to migrated positions
        particles = new_particles

    return global_best_position, global_best_value
