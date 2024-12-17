import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt


# -------------------------- helper functions --------------------------
# Adaptive Crossover function to gradually decrease the crossover probability
def get_adaptive_crossover_prob(current_gen, total_gen, max_cxpb=0.9, min_cxpb=0.6):
    return max_cxpb - ((max_cxpb - min_cxpb) * (current_gen / total_gen))


def get_adaptive_mutation_prob(current_gen, total_gen, min_mutpb=0.2, max_mutpb=0.5):
    return min_mutpb + ((max_mutpb - min_mutpb) * (current_gen / total_gen))


def calculate_population_diversity(population):
    unique_individuals = len(set(map(tuple, population)))  # Counting unique individuals
    total_population = len(population)
    diversity = unique_individuals / total_population
    return diversity


# dataset preparation for fitness function
X = np.load('encoded_text.npy')  # Load encoded features from .npy file
df = pd.read_csv('dataset/processed_sentiment_data.csv')  # Load the sentiment labels
y = df['sentiment']  # Load binaries (target class)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # train test split

# -------------------------- DEAP Setup --------------------------
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))  # to maximize all metrics
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()
n_features = X_train.shape[1]  # retrieving genome size, which should be 384
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Fitness function
def fitness_function(individual):
    # Select features based on the binary coded individual
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if not selected_features:  # Handling case where no features are selected
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

# Static Parameters
population_size = 100
num_generations = 100

# Creating the initial population
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


# -------------------------- Evolution process --------------------------
def eaMuPlusLambdaWithAdaptiveMutation(population, toolbox, mu, lambda_, ngen, stats=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    diversity_values = []

    # Evaluating the initial population
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Updating HoF initially
    if hof is not None:
        hof.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Logging initial diversity
    diversity = calculate_population_diversity(population)
    diversity_values.append(diversity)

    # Generational process
    for gen in range(1, ngen + 1):
        # Calculating population diversity
        diversity = calculate_population_diversity(population)
        diversity_values.append(diversity)

        # Dynamically calculating crossover and mutation probability
        current_mutpb = get_adaptive_mutation_prob(gen, ngen)
        current_cxpb = get_adaptive_crossover_prob(gen, ngen)

        # Computing Pareto fronts and crowding distances for the use in TournamentDCD
        fronts = tools.sortNondominated(population, k=len(population), first_front_only=False)
        for front in fronts:
            tools.emo.assignCrowdingDist(front)

        # Select the mating pool using selTournamentDCD
        offspring = toolbox.tournament(population, lambda_)

        # Apply variation (crossover and mutation)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=current_cxpb, mutpb=current_mutpb)

        # Evaluating offspring fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Updating Hall of Fame with both offspring and population
        if hof is not None:
            hof.update(offspring + population)

        population[:] = tools.selNSGA2(population + offspring, mu)

        # logging statistics in current generation
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), mutpb=current_mutpb, cxpb=current_cxpb, **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, diversity_values


# calling evolution process
result_population, logbook, diversity_values = eaMuPlusLambdaWithAdaptiveMutation(
    population,
    toolbox,
    mu=population_size,
    lambda_=population_size,
    ngen=num_generations,
    stats=multi_stats,
    verbose=True,
)

# Accessing F1, Accuracy, and AUC over generations for plotting
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


# Plot Diversity Over Generations
plt.figure(figsize=(12, 4))
plt.plot(generations, diversity_values, label="Diversity", color='purple', linestyle='-', marker='o', alpha=0.7)
plt.xlabel("Generation")
plt.ylabel("Diversity")
plt.title("Diversity Over Generations")
plt.legend()
plt.grid(True)
plt.show()


# Displaying Pareto Front
print("\nPareto Front:")
for ind in hof:
    print(f"Selected Features: {sum(ind)}")
    print(f"F1 Score: {ind.fitness.values[0]:.4f}, Accuracy: {ind.fitness.values[1]:.4f}, AUC: {ind.fitness.values[2]:.4f}")

# Retrieving the Best Individual Based on Accuracy
best_individual = max(hof, key=lambda ind: ind.fitness.values[1])
selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
print(f"\nBest Individual for F1 Score: {selected_features}")
print("Selected Features:", len(selected_features))
print("Fitness Values (F1, Accuracy, AUC):", best_individual.fitness.values)

# -------------------------- Evaluation with SVM --------------------------
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

print(f"================== Trained SVM Model Output ==================")
print(f"F1 Score: {svm_f1:.4f}")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"AUC: {svm_auc:.4f}")
