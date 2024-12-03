import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


# Helper functions
def get_adaptive_mutation_prob(current_gen, total_gen, min_mutpb=0.2, max_mutpb=0.5):
    return min_mutpb + ((max_mutpb - min_mutpb) * (current_gen / total_gen))


# Adaptive Crossover function to gradually decrease the crossover probability
def get_adaptive_crossover_prob(current_gen, total_gen, max_cxpb=0.9, min_cxpb=0.6):
    return max_cxpb - ((max_cxpb - min_cxpb) * (current_gen / total_gen))


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

# Parameters
population_size = 100
num_generations = 100
crossover_prob = 0.7
mutation_prob = 0.2

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
def eaMuPlusLambdaWithAdaptiveMutation(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None,
                                       verbose=__debug__):
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

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Dynamically calculate mutation probability for this generation
        current_mutpb = get_adaptive_mutation_prob(gen, ngen)
        current_cxpb = get_adaptive_crossover_prob(gen, ngen)

        # Select the next generation individuals
        offspring = toolbox.select(population, lambda_)

        # Vary the offspring with adaptive mutation and crossover
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=current_cxpb, mutpb=current_mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Ensure Hall of Fame is updated after each generation
        if hof is not None:
            hof.update(offspring)

        # Combine population and offspring for the next generation
        population[:] = tools.selNSGA2(offspring + population, mu)

        # Compile statistics
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), mutpb=current_mutpb, **record)
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
        cxpb=crossover_prob,
        mutpb=mutation_prob,
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

# Plot Max and Mean F1 Scores for all iterations
plt.figure(figsize=(12, 4))
for i in range(num_iterations):
    plt.plot(generations_all_iterations[0], f1_max_all_iterations[i], label=f"Max F1 Iter {i + 1}")
    plt.plot(generations_all_iterations[0], f1_mean_all_iterations[i], label=f"Mean F1 Iter {i + 1}", linestyle='--')
plt.xlabel("Generation")
plt.ylabel("F1 Score")
plt.title("F1 Score Over Generations (Multiple Iterations)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Max and Mean Accuracy Scores for all iterations
plt.figure(figsize=(12, 4))
for i in range(num_iterations):
    plt.plot(generations_all_iterations[0], accuracy_max_all_iterations[i], label=f"Max Accuracy Iter {i + 1}")
    plt.plot(generations_all_iterations[0], accuracy_mean_all_iterations[i], label=f"Mean Accuracy Iter {i + 1}",
             linestyle='--')
plt.xlabel("Generation")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Generations (Multiple Iterations)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Max and Mean AUC Scores for all iterations
plt.figure(figsize=(12, 4))
for i in range(num_iterations):
    plt.plot(generations_all_iterations[0], auc_max_all_iterations[i], label=f"Max AUC Iter {i + 1}")
    plt.plot(generations_all_iterations[0], auc_mean_all_iterations[i], label=f"Mean AUC Iter {i + 1}", linestyle='--')
plt.xlabel("Generation")
plt.ylabel("AUC")
plt.title("AUC Over Generations (Multiple Iterations)")
plt.legend()
plt.grid(True)
plt.show()
