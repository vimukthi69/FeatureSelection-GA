import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt


# helper functions
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

    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)[:, 1]

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
mutation_probability = 0.3

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


def eaMuPlusLambdaWithAdaptiveMutation(population, toolbox, mu, lambda_, mutpb, ngen, stats=None, verbose=__debug__):
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
        current_cxpb = get_adaptive_crossover_prob(gen, ngen)

        # Compute Pareto fronts and crowding distances
        fronts = tools.sortNondominated(population, k=len(population), first_front_only=False)
        for front in fronts:
            tools.emo.assignCrowdingDist(front)

        # Select the mating pool using selTournamentDCD
        offspring = toolbox.tournament(population, lambda_)

        # Apply variation (crossover and mutation)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=current_cxpb, mutpb=mutpb)

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
        logbook.record(gen=gen, nevals=len(invalid_ind), mutpb=mutpb, cxpb=current_cxpb, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


# Run the updated NSGA-II with adaptive mutation
result_population, logbook = eaMuPlusLambdaWithAdaptiveMutation(
    population,
    toolbox,
    mu=population_size,
    lambda_=population_size,
    mutpb=mutation_probability,
    ngen=num_generations,
    stats=multi_stats,
    verbose=True,
)

# Plot F1, Accuracy, and AUC over generations
generations = logbook.select("gen")
f1_max = logbook.chapters["F1"].select("max")
f1_mean = logbook.chapters["F1"].select("mean")
accuracy_max = logbook.chapters["Accuracy"].select("max")
accuracy_mean = logbook.chapters["Accuracy"].select("mean")
auc_max = logbook.chapters["AUC"].select("max")
auc_mean = logbook.chapters["AUC"].select("mean")

# Plot F1 Score
plt.figure(figsize=(12, 4))
plt.plot(generations, f1_max, label="Max F1 Score", linestyle='-', color='b', markersize=4, marker='o', alpha=0.7)
plt.plot(generations, f1_mean, label="Mean F1 Score", linestyle='--', color='g', markersize=4, marker='x', alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("F1 Score")
plt.title("F1 Score Over Generations")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(12, 4))
plt.plot(generations, accuracy_max, label="Max Accuracy", linestyle='-', color='b', markersize=4, marker='o', alpha=0.7)
plt.plot(generations, accuracy_mean, label="Mean Accuracy", linestyle='--', color='g', markersize=4, marker='x', alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Generations")
plt.legend()
plt.grid(True)
plt.show()

# Plot AUC
plt.figure(figsize=(12, 4))
plt.plot(generations, auc_max, label="Max AUC", linestyle='-', color='b', markersize=4, marker='o', alpha=0.7)
plt.plot(generations, auc_mean, label="Mean AUC", linestyle='--', color='g', markersize=4, marker='x', alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("AUC")
plt.title("AUC Over Generations")
plt.legend()
plt.grid(True)
plt.show()


# Display Pareto Front
print("\nPareto Front:")
for ind in hof:
    print(f"Selected Features: {sum(ind)}")
    print(f"F1 Score: {ind.fitness.values[0]:.4f}, Accuracy: {ind.fitness.values[1]:.4f}, AUC: {ind.fitness.values[2]:.4f}")

# Retrieve the Best Individual Based on F1 Score
best_individual = max(hof, key=lambda ind: ind.fitness.values[0])  # Maximize F1
selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
print("\nBest Individual for F1 Score:")
print("Selected Features:", len(selected_features))
print("Fitness Values (F1, Accuracy, AUC):", best_individual.fitness.values)

# Final Evaluation with SVM
print("\nEvaluating SVM with Selected Features:")
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Train SVM on Selected Features
svm_model = SVC(probability=True, kernel="linear", random_state=42)  # Linear kernel for simplicity
svm_model.fit(X_train_selected, y_train)

# Predict and Evaluate
y_pred_svm = svm_model.predict(X_test_selected)
y_proba_svm = svm_model.predict_proba(X_test_selected)[:, 1]
svm_f1 = f1_score(y_test, y_pred_svm)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_auc = roc_auc_score(y_test, y_proba_svm)

print(f"SVM Metrics:")
print(f"F1 Score: {svm_f1:.4f}")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"AUC: {svm_auc:.4f}")
