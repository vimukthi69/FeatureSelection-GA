import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt


# Helper functions
def get_adaptive_mutation_prob(current_gen, total_gen, min_mutpb=0.2, max_mutpb=0.5):
    return min_mutpb + ((max_mutpb - min_mutpb) * (current_gen / total_gen))


# Adaptive Crossover function to gradually decrease the crossover probability
def get_adaptive_crossover_prob(current_gen, total_gen, max_cxpb=0.9, min_cxpb=0.6):
    return max_cxpb - ((max_cxpb - min_cxpb) * (current_gen / total_gen))


def calculate_population_diversity(population):
    unique_individuals = len(set(map(tuple, population)))  # Counting unique individuals
    total_population = len(population)
    diversity = unique_individuals / total_population
    return diversity


def restart_population(population, toolbox, proportion):
    size = int(len(population) * proportion)
    print(f"Triggering population restart: Replacing {size} individuals.")

    # Replace the specified proportion with new individuals
    for i in range(size):
        population[-(i + 1)] = toolbox.individual()

    # Evaluate the new individuals to assign fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


def probabilistic_elitism(population, elite_ratio=0.1, retain_prob=0.7):
    elite_size = int(len(population) * elite_ratio)
    parents_sorted = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    elite_parents = parents_sorted[:elite_size]  # Top elites

    # Retain elites probabilistically
    retained_elites = [ind for ind in elite_parents if random.random() < retain_prob]
    print(f"Retaining {len(retained_elites)} elite individuals probabilistically.")
    return retained_elites


# Load encoded features from .npy file
X = np.load('encoded_text.npy')

# Load the sentiment labels
df = pd.read_csv('dataset/processed_sentiment_data.csv')
y = df['sentiment']  # Convert to binary

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DEAP Setup
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))  # Maximize all metrics
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()

# Define genome: Binary vector of length equal to the number of features (e.g., 384)
n_features = X_train.shape[1]
toolbox.register("attr_bool", random.randint, 0, 1)  # Each bit is 0 or 1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Fitness function: Evaluate F1 score, Accuracy, and AUC
def fitness_function(individual):
    # Select features based on the binary vector (genome)
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if not selected_features:  # Handle case where no features are selected
        return 0.0, 0.0, 0.0  # Return all metrics as 0.0

    # Subset the training and test data
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_selected, y_train)

    # Predict and calculate
    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)[:, 1]  # Probabilities for AUC calculation

    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return f1, accuracy, auc


toolbox.register("evaluate", fitness_function)

# Genetic operators
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)
toolbox.register("tournament", tools.selTournamentDCD)

# Parameters
population_size = 100
num_generations = 100

# Create the initial population
population = toolbox.population(n=population_size)

# Statistics for F1, Accuracy, and AUC
stats_f1 = tools.Statistics(lambda ind: ind.fitness.values[0])
stats_f1.register("max", np.max)
stats_f1.register("mean", np.mean)
stats_accuracy = tools.Statistics(lambda ind: ind.fitness.values[1])
stats_accuracy.register("max", np.max)
stats_accuracy.register("mean", np.mean)
stats_auc = tools.Statistics(lambda ind: ind.fitness.values[2])
stats_auc.register("max", np.max)
stats_auc.register("mean", np.mean)
multi_stats = tools.MultiStatistics(F1=stats_f1, Accuracy=stats_accuracy, AUC=stats_auc)

# Hall of Fame for the Pareto Front
hof = tools.ParetoFront()


# Define the GA with adaptive mutation and crossover
def eaMuPlusLambdaWithAdaptiveMutation(population, toolbox, mu, lambda_, ngen, stats=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Ensure Hall of Fame is updated
    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    stagnant_generations = 0
    max_stagnant_generations = 3

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Calculate population diversity
        diversity = calculate_population_diversity(population)
        print(f"Generation {gen}: Diversity = {diversity:.2f}")

        # Check for stagnant diversity
        if diversity < 0.9:
            stagnant_generations += 1
        else:
            stagnant_generations = 0

        # Trigger population restart if stagnation persists
        if stagnant_generations >= max_stagnant_generations:
            restart_population(population, toolbox, proportion=0.8)
            stagnant_generations = 0

        # Dynamically calculate mutation probability for this generation
        current_mutpb = get_adaptive_mutation_prob(gen, ngen)
        current_cxpb = get_adaptive_crossover_prob(gen, ngen)

        # Compute Pareto fronts and crowding distances
        fronts = tools.sortNondominated(population, k=len(population), first_front_only=False)
        for front in fronts:
            tools.emo.assignCrowdingDist(front)

        # Select the mating pool using selTournamentDCD
        offspring = toolbox.tournament(population, lambda_)

        # Apply variation (crossover and mutation)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=current_cxpb, mutpb=current_mutpb)

        # Evaluate offspring fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update Hall of Fame with both offspring and population
        if hof is not None:
            hof.update(offspring + population)

        # Retain probabilistic elites
        retained_elites = probabilistic_elitism(population, elite_ratio=0.1, retain_prob=0.7)
        combined_population = retained_elites + offspring

        # Select the next generation using NSGA-II
        population[:] = tools.selNSGA2(combined_population, mu)

        # Compile and log statistics
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), mutpb=current_mutpb, cxpb=current_cxpb, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# Run the GA for 4 iterations
num_iterations = 4

# Lists to store results for plotting (for all iterations)
f1_max_all_iterations = []
f1_mean_all_iterations = []
accuracy_max_all_iterations = []
accuracy_mean_all_iterations = []
auc_max_all_iterations = []
auc_mean_all_iterations = []

# List to store generations for consistency across iterations
generations_all_iterations = []

for iteration in range(num_iterations):
    print(f"Running iteration {iteration + 1} of {num_iterations}...")

    # Reinitialize the population and Hall of Fame at the start of each iteration
    population = toolbox.population(n=population_size)
    hof.clear()

    # Run the GA with adaptive mutation
    result_population, logbook = eaMuPlusLambdaWithAdaptiveMutation(
        population,
        toolbox,
        mu=population_size,
        lambda_=population_size,
        ngen=num_generations,
        stats=multi_stats,
        verbose=True,
    )

    # Collect F1, Accuracy, and AUC max and mean values for this iteration
    generations = logbook.select("gen")
    f1_max = logbook.chapters["F1"].select("max")
    f1_mean = logbook.chapters["F1"].select("mean")
    accuracy_max = logbook.chapters["Accuracy"].select("max")
    accuracy_mean = logbook.chapters["Accuracy"].select("mean")
    auc_max = logbook.chapters["AUC"].select("max")
    auc_mean = logbook.chapters["AUC"].select("mean")

    # Store generations from the first iteration for consistency
    if iteration == 0:
        generations_all_iterations.append(generations)

    # Append results from this iteration to the corresponding lists
    f1_max_all_iterations.append(f1_max)
    f1_mean_all_iterations.append(f1_mean)
    accuracy_max_all_iterations.append(accuracy_max)
    accuracy_mean_all_iterations.append(accuracy_mean)
    auc_max_all_iterations.append(auc_max)
    auc_mean_all_iterations.append(auc_mean)

# Plot Max F1 Score Over Generations for each iteration
plt.figure(figsize=(12, 4))
for i in range(num_iterations):
    plt.plot(generations_all_iterations[0], f1_max_all_iterations[i], label=f"Max F1 Iter {i+1}", marker='o', markersize=6, linestyle='-', alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Max F1 Score")
plt.title("Max F1 Score Over Generations (Multiple Iterations)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Mean F1 Score Over Generations for each iteration
plt.figure(figsize=(12, 4))
for i in range(num_iterations):
    plt.plot(generations_all_iterations[0], f1_mean_all_iterations[i], label=f"Mean F1 Iter {i+1}", marker='x', markersize=6, linestyle='--', alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Mean F1 Score")
plt.title("Mean F1 Score Over Generations (Multiple Iterations)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Max Accuracy Over Generations for each iteration
plt.figure(figsize=(12, 4))
for i in range(num_iterations):
    plt.plot(generations_all_iterations[0], accuracy_max_all_iterations[i], label=f"Max Accuracy Iter {i+1}", marker='o', markersize=6, linestyle='-', alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Max Accuracy")
plt.title("Max Accuracy Over Generations (Multiple Iterations)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Mean Accuracy Over Generations for each iteration
plt.figure(figsize=(12, 4))
for i in range(num_iterations):
    plt.plot(generations_all_iterations[0], accuracy_mean_all_iterations[i], label=f"Mean Accuracy Iter {i+1}", marker='x', markersize=6, linestyle='--', alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Mean Accuracy")
plt.title("Mean Accuracy Over Generations (Multiple Iterations)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Max AUC Over Generations for each iteration
plt.figure(figsize=(12, 4))
for i in range(num_iterations):
    plt.plot(generations_all_iterations[0], auc_max_all_iterations[i], label=f"Max AUC Iter {i+1}", marker='o', markersize=6, linestyle='-', alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Max AUC")
plt.title("Max AUC Over Generations (Multiple Iterations)")
plt.legend()
plt.grid(True)
plt.show()