import neat
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import classification_report, f1_score
import visualize


df = pd.read_csv("Data/StressAppraisal.csv")
data = df.drop(columns=["Productivity", "Mood", "Stress_Numeric", "Stress"])
labels = df["Stress"]
labels_encoded, uniques = pd.factorize(labels) # 0 - Boredom, 1 - Distress, 2 - Eustress, 3 - Eustress-distress coexistence
data_train, data_test, labels_train, labels_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=17, stratify=labels_encoded)
num_train_samples = len(data_train)

# NORMALIZE FEATURES

# params
num_generations = 100

# functions for neat
def fitness_function(net):
    num_correct_predictions = 0
    for curX, curY in zip(data_train.values, labels_train):
        net_output = net.activate(curX)
        prediction = np.argmax(net_output)
        if prediction == curY:
            num_correct_predictions += 1
    return num_correct_predictions/num_train_samples


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = fitness_function(net)


# might need to alter this config
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'neat-config')

population = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(5))

# Run for up to 300 generations.
winner = population.run(eval_genomes, num_generations)

# Display the winning genome.
print(f'\nBest genome:\n{winner!s}')


# this needs to be altered but find a way to do something like this for the stress data
# # Show output of the most fit genome against training data.
# print('\nOutput:')
# winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
# for xi, xo in zip(xor_inputs, xor_outputs):
#     output = winner_net.activate(xi)
#     print(f"input {xi!r}, expected output {xo!r}, got {output!r}")

# node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
# visualize.draw_net(config, winner, True, node_names=node_names)
# visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
# visualize.plot_stats(stats, ylog=False, view=True)
# visualize.plot_species(stats, view=True)

# p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
# p.run(eval_genomes, 10)