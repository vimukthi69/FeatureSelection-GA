import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


# helper functions
def calculate_diversity(population):
    """
    Calculate diversity in the population based on pairwise Hamming distance.
    :param population: List of individuals (binary genomes).
    :return: Average Hamming distance and proportion of active genes.
    """
    # Convert population to a binary matrix
    genome_matrix = np.array(population)

    # Pairwise Hamming distances
    hamming_distances = pdist(genome_matrix, metric="hamming")
    avg_hamming_distance = np.mean(hamming_distances)

    # Proportion of active genes (1s) in the genome
    active_genes_ratio = np.mean(genome_matrix)

    return avg_hamming_distance, active_genes_ratio


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
    # Train an SVM model
    # model = SVC(probability=True)  # Enable probability predictions for AUC
    # model.fit(X_train_selected, y_train)

    # Predict and calculate metrics
    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)[:, 1]  # Probabilities for AUC calculation

    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    return f1, accuracy, auc


toolbox.register("evaluate", fitness_function)

# Genetic operators
toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Bit-flip mutation
toolbox.register("select", tools.selNSGA2)   # NSGA-II selection

# Parameters
population_size = 50
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

# Store diversity metrics over generations
diversity_hamming = []
diversity_active_genes = []

# Run NSGA-II
result_population, logbook = algorithms.eaMuPlusLambda(
    population,
    toolbox,
    mu=population_size,
    lambda_=population_size,
    cxpb=crossover_prob,
    mutpb=mutation_prob,
    ngen=num_generations,
    stats=multi_stats,
    halloffame=hof,
    verbose=True,
)

# Calculate diversity at each generation
for gen in range(num_generations + 1):  # +1 because generations start from 0
    avg_hamming_distance, active_genes_ratio = calculate_diversity(population)
    diversity_hamming.append(avg_hamming_distance)
    diversity_active_genes.append(active_genes_ratio)

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
plt.plot(generations, f1_max, label="Max F1 Score", marker="o")
plt.plot(generations, f1_mean, label="Mean F1 Score", marker="x")
plt.xlabel("Generation")
plt.ylabel("F1 Score")
plt.title("F1 Score Over Generations")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(12, 4))
plt.plot(generations, accuracy_max, label="Max Accuracy", marker="o")
plt.plot(generations, accuracy_mean, label="Mean Accuracy", marker="x")
plt.xlabel("Generation")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Generations")
plt.legend()
plt.grid(True)
plt.show()

# Plot AUC
plt.figure(figsize=(12, 4))
plt.plot(generations, auc_max, label="Max AUC", marker="o")
plt.plot(generations, auc_mean, label="Mean AUC", marker="x")
plt.xlabel("Generation")
plt.ylabel("AUC")
plt.title("AUC Over Generations")
plt.legend()
plt.grid(True)
plt.show()

# Plot Hamming Distance Diversity
plt.figure(figsize=(12, 4))
plt.plot(range(num_generations + 1), diversity_hamming, label="Hamming Distance", marker="o")
plt.xlabel("Generation")
plt.ylabel("Average Hamming Distance")
plt.title("Population Diversity (Hamming Distance) Over Generations")
plt.grid(True)
plt.legend()
plt.show()

# Plot Active Genes Diversity
plt.figure(figsize=(12, 4))
plt.plot(range(num_generations + 1), diversity_active_genes, label="Active Genes Ratio", marker="x", color="orange")
plt.xlabel("Generation")
plt.ylabel("Active Genes Ratio")
plt.title("Proportion of Active Genes Over Generations")
plt.grid(True)
plt.legend()
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
