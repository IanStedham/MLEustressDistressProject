import neat
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import visualize
import os
import pickle

df = pd.read_csv("Data/StressAppraisal.csv")
data = df.drop(columns=["Productivity", "Mood", "Stress_Numeric", "Stress"])
labels = df["Stress"]
labels_encoded, uniques = pd.factorize(labels) # 0 - Boredom, 1 - Distress, 2 - Eustress, 3 - Eustress-distress coexistence
data_train, data_test, labels_train, labels_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=17, stratify=labels_encoded)

print(f"Number of features: {data_train.shape[1]}")
print(f"Number of classes: {len(uniques)}")
print(f"Class mapping: {dict(enumerate(uniques))}")

print("Original class distribution:")
unique_orig, counts_orig = np.unique(labels_encoded, return_counts=True)
for cls, count in zip(unique_orig, counts_orig):
    print(f"  {uniques[cls]}: {count} ({count/len(labels_encoded)*100:.1f}%)")

data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels_encoded, test_size=0.2, random_state=17, stratify=labels_encoded
)

data_train_sub, data_val, labels_train_sub, labels_val = train_test_split(
    data_train, labels_train, test_size=0.15, random_state=17, stratify=labels_train
)

print(f"\nData split:")
print(f"  Training: {len(labels_train_sub)}")
print(f"  Validation: {len(labels_val)}")
print(f"  Test: {len(labels_test)}")

# Apply SMOTE to training data before scaling
smote = SMOTE(random_state=17, k_neighbors=5)
data_train_smote, labels_train_smote = smote.fit_resample(data_train_sub, labels_train_sub)

print("\nTraining set class distribution after SMOTE:")
unique_train, counts_train = np.unique(labels_train_smote, return_counts=True)
for cls, count in zip(unique_train, counts_train):
    print(f"  {uniques[cls]}: {count} ({count/len(labels_train_smote)*100:.1f}%)")


# NORMALIZE FEATURES
scaler = StandardScaler()
training_features = scaler.fit_transform(data_train_smote)
validation_features = scaler.transform(data_val.to_numpy())
testing_features = scaler.transform(data_test.values)
num_train_samples = len(training_features)

print(f"Feature means after scaling: {training_features.mean(axis=0).round(3)}")
print(f"Feature stds after scaling: {training_features.std(axis=0).round(3)}")


# params
num_generations = 500
validation_check_interval = 20
best_val_f1 = 0
no_improvement_count = 0
patience = 60

def evaluate_network(net, features, labels):
    """Evaluate network and return metrics."""
    predictions = [np.argmax(net.activate(xi)) for xi in features]
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    return predictions, accuracy, f1_macro, f1_weighted

def test_best_network(genome, config):
    """Test the best genome on test data."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    predictions, accuracy, f1_macro, f1_weighted = evaluate_network(
        net, testing_features, labels_test
    )
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(labels_test, predictions, target_names=uniques, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels_test, predictions))
    
    # Show prediction distribution
    unique_pred, counts_pred = np.unique(predictions, return_counts=True)
    print("\nPrediction distribution:")
    for cls in range(4):
        count = counts_pred[unique_pred == cls][0] if cls in unique_pred else 0
        print(f"  {uniques[cls]:30s}: {count:3d} ({count/len(predictions)*100:.1f}%)")
    
    return net, predictions, accuracy, f1_macro

# functions for neat, if this goeas poorly try the simply just f1 and accuracy with no bonuses or penalties 
def fitness_function(net):
    predictions = []
    true_labels = []
    
    for xi, yi in zip(training_features, labels_train_smote):
        output = net.activate(xi)
        predictions.append(np.argmax(output))
        true_labels.append(yi)
    
    # Base fitness: Macro F1
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(true_labels, predictions)
    base_fitness = 0.5 * f1_macro + 0.5 * accuracy
    
    return base_fitness


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = fitness_function(net)


class ValidationReporter(neat.reporting.BaseReporter):
    """Reporter with validation monitoring."""
    
    def __init__(self, validation_interval=20):
        self.generation = 0
        self.validation_interval = validation_interval
        self.best_val_fitness = 0
        self.no_improvement = 0
        
    def start_generation(self, generation):
        self.generation = generation
        print(f'\nGeneration {generation}', end='')
        
    def post_evaluate(self, config, population, species, best_genome):
        global best_val_fitness, no_improvement_count
        
        # Show training fitness
        print(f' | Best fitness: {best_genome.fitness:.4f}')
        
        # Validation check
        if self.generation % self.validation_interval == 0 and self.generation > 0:
            net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            _, val_acc, val_f1, val_f1_w = evaluate_network(
                net, validation_features, labels_val
            )
            
            print(f'\n{"="*60}')
            print(f'Validation Check - Generation {self.generation}')
            print(f'{"="*60}')
            print(f'Training F1:    {best_genome.fitness:.4f}')
            print(f'Validation Acc: {val_acc:.4f}')
            print(f'Validation F1:  {val_f1:.4f} (macro) | {val_f1_w:.4f} (weighted)')
            print(f'Gap:            {(best_genome.fitness - val_f1):.4f}')
            
            # Track best
            if (val_f1*0.5 + val_acc*0.5) > self.best_val_fitness:
                improvement = (val_f1*0.5 + val_acc*0.5) - self.best_val_fitness
                self.best_val_fitness = (val_f1*0.5 + val_acc*0.5)
                best_val_fitness = (val_f1*0.5 + val_acc*0.5)
                self.no_improvement = 0
                no_improvement_count = 0
                print(f'✓ New best validation Fitness! (+{improvement:.4f})')
                
                # Save best validation model
                with open('best_validation_genome.pkl', 'wb') as f:
                    pickle.dump(best_genome, f)
            else:
                self.no_improvement += self.validation_interval
                no_improvement_count += self.validation_interval
                print(f'⚠ No improvement for {self.no_improvement} generations')
            
            # Overfitting warning
            if best_genome.fitness - (val_f1*0.5 + val_acc*0.5) > 0.15:
                print(f'⚠️ WARNING: Overfitting detected!')
            
            print(f'{"="*60}')
            
            # Early stopping
            if self.no_improvement >= patience:
                print(f'\nEARLY STOPPING at generation {self.generation}')
                print(f'Best validation F1: {self.best_val_fitness:.4f}')
                return True
        
        return False

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat_config')

population = neat.Population(config)

population.add_reporter(ValidationReporter(validation_interval=validation_check_interval))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

# Run evolution
print(f"\n{'='*60}")
print("Starting NEAT Evolution")
print(f"{'='*60}")
print(f"Generations: {num_generations}")
print(f"Population: {config.pop_size}")
print(f"Validation checks every {validation_check_interval} generations")
print(f"Early stopping patience: {patience} generations")
print(f"{'='*60}\n")

# Run for up to 300 generations.
winner = population.run(eval_genomes, num_generations)

# Display the winning genome.
print(f'\nBest genome:\n{winner!s}')
print("\n" + "="*50)
print("BEST GENOME")
print("="*50)
print(f"Fitness: {winner.fitness:.4f}")
print(f"Number of nodes: {len(winner.nodes)}")
print(f"Number of connections: {len(winner.connections)}")

best_net, predictions, test_acc, test_f1 = test_best_network(winner, config)

with open('final_genome.pkl', 'wb') as f:
    pickle.dump(winner, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('stats.pkl', 'wb') as f:
    pickle.dump(stats, f)
