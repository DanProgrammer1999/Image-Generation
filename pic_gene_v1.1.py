# In[1]:

from timeit import default_timer as timer

import numpy as np
import cv2 as cv
from skimage.measure import compare_mse


class Figures:
    CIRCLE = 0
    LINE = 1
    POLYGON = 2
    RANDOM = 3


class InitialColor:
    BLACK = 0
    WHITE = 1
    FROM_CHECKPOINT = 2


class Parameters:

    # General Parameters
    population_size = 5
    number_of_generations = 500000
    initial_population_size = 5
    initial_color = InitialColor.FROM_CHECKPOINT

    # Input and Output
    # picture_name = 'Lisa.jpg'
    # dir_out = 'progress/2'

    picture_name = 'florence.jpg'
    dir_in = '.'
    dir_out = 'progress/3'
    checkpoint_mod = 100

    save_mod = 1000

    # Start from some picture (so that we can stop execution and then continue from checkpoint)
    checkpoint = '{}/checkpoint.jpg'.format(dir_out)

    # Circle
    min_radius = 1
    max_radius = 5
    # Polygons
    bounding_box_size = 10
    n_vertices = 3
    # Line
    min_line_length = 5
    max_line_length = 10

    figure = Figures.RANDOM

    # Mutation
    mutation_chance = 1
    mutation_amount = 3

    # Crossover
    crossover_percent = 0.6

    # Selection
    selection_percent = 0.2


class Circle:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def draw(self, picture):
        cv.circle(picture, self.center, self.radius, self.color, thickness=cv.FILLED)

    @staticmethod
    def get_random():
        radius = np.random.randint(low=Parameters.min_radius, high=Parameters.max_radius)

        x = np.random.randint(0, Utils.pic.shape[0])
        y = np.random.randint(0, Utils.pic.shape[1])
        color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))

        return Circle((x, y), radius, color)


class Line:
    def __init__(self, point1, point2, color):
        self.point1 = point1
        self.point2 = point2
        self.color = color

    def draw(self, picture):
        cv.line(picture, self.point1, self.point2, self.color)

    @staticmethod
    def get_random():
        [x1, y1] = np.random.randint(0, Utils.pic.shape[0] - Parameters.min_line_length, 2)
        x2 = np.random.randint(x1 + Parameters.min_line_length, x1 + Parameters.max_line_length)

        del_x = x2 - x1
        lower_boundary = int(np.sqrt(Parameters.min_line_length**2 - del_x**2) + y1) \
            if Parameters.min_line_length**2 > del_x**2 else y1

        upper_boundary = int(np.sqrt(Parameters.max_line_length**2 - (x2 - x1)**2) + y1)
        y2 = np.random.randint(lower_boundary, upper_boundary)

        color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))

        return Line((x1, y1), (x2, y2), color)


class Polygon:
    def __init__(self, points, color):
        self.points = points
        self.color = color

    def draw(self, picture):
        points = self.points
        cv.fillPoly(picture, [points], self.color)

    @staticmethod
    def get_random():
        curr_point = [np.random.uniform(0, Utils.pic.shape[0]), np.random.uniform(0, Utils.pic.shape[1])]
        size = [10, 10]
        points = []
        for i in range(Parameters.n_vertices):

            point_x = int(np.clip(curr_point[0] + np.random.uniform(-1, 1)*size[0], 0, Utils.pic.shape[0]))
            point_y = int(np.clip(curr_point[1] + np.random.uniform(-1, 1)*size[1], 0, Utils.pic.shape[1]))
            points.append([point_x, point_y])

        color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))

        return Polygon(np.array(points, dtype=np.int32), color)


# Chromosome class
class Chromosome:
    def __init__(self, genes=None):
        if genes is None:
            genes = Utils.get_initial()

        self.genes = genes
        self.score = Genetics.fitness_score(self.genes)
        self.max_score = self.score

    def mutate(self, chance=Parameters.mutation_chance):
        choice = np.random.uniform()
        if choice >= chance:
            return

        for _ in range(Parameters.mutation_amount):

            figure = None
            selection = Parameters.figure

            if Parameters.figure == Figures.RANDOM:
                all_attr = vars(Figures).keys()
                num_options = sum([1 if not key.startswith('__') else 0 for key in all_attr]) - 1
                selection = np.random.randint(num_options)

            if selection == Figures.POLYGON:
                figure = Polygon.get_random()

            if selection == Figures.CIRCLE:
                figure = Circle.get_random()

            if selection == Figures.LINE:
                figure = Line.get_random()

            new_picture = self.genes.copy()
            figure.draw(new_picture)
            new_score = Genetics.fitness_score(new_picture)

            if new_score < self.score:
                self.genes = new_picture
                self.score = new_score

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return str(self.score)


# Utilities for the genetic algorithm
class Utils:

    pic = cv.imread(Parameters.picture_name)

    @staticmethod
    def track_time(func, name=None, *args):
        if name is None:
            name = func.__name__

        start = timer()
        result = func(*args)
        end = timer()

        print(name, 'elapsed', str(end - start), 'seconds')

        return result

    @staticmethod
    def get_initial():
        result = []
        if Parameters.initial_color == InitialColor.BLACK:
            result = np.zeros_like(Utils.pic)

        if Parameters.initial_color == InitialColor.WHITE:
            result = np.ones_like(Utils.pic)*255

        if Parameters.initial_color == InitialColor.FROM_CHECKPOINT:
            result = cv.imread(Parameters.checkpoint)
            if result is None:
                print("Could not load from chekpoint")
                result = np.zeros_like(Utils.pic)

        return result


class Genetics:

    np_mutation = np.vectorize(Chromosome.mutate)

    @staticmethod
    def cross2(sample1, sample2):
        weight1_rel = sample1.score
        weight2_rel = sample2.score
        weight1 = weight1_rel/(weight1_rel + weight2_rel)
        weight2 = weight2_rel/(weight1_rel + weight2_rel)

        res = cv.addWeighted(sample1.genes, weight1, sample2.genes, weight2, 0)

        return Chromosome(res)

    @staticmethod
    def crossover(generation):
        cross_number = int(Parameters.crossover_percent*len(generation))

        res = []
        for i in range(1, cross_number):
            child = Genetics.cross2(generation[i - 1], generation[i])
            res.append(child)

        return res

    @staticmethod
    def selection(generation):
        n_remove = int(Parameters.selection_percent*len(generation))
        return generation[: len(generation) - n_remove]

    @staticmethod
    def fitness_score(current, original=Utils.pic):
        return compare_mse(current, original)

    @staticmethod
    def evolution(verbose=True):

        generation = np.array([Chromosome() for _ in range(Parameters.initial_population_size)])
        best_picture = generation[0].genes
        best_score = generation[0].score
        max_score = Genetics.fitness_score(np.zeros_like(Utils.pic), Utils.pic)

        for i in range(Parameters.number_of_generations):

            if verbose:
                print("\n------------------------\nIteration {}\n".format(i))
                children = Utils.track_time(Genetics.crossover, 'Crossover', generation)
                generation = np.concatenate((generation, children))
                # print('After Crossover:', len(generation))

                Utils.track_time(Genetics.np_mutation, 'Mutation', generation)

                generation.sort()
                generation = Genetics.selection(generation)
                # print('After Selection:', len(generation))

                if generation[0].score < best_score:
                    best_score = generation[0].score
                    best_picture = generation[0].genes

                if len(generation) > Parameters.population_size:
                    generation = generation[:Parameters.population_size]

                if i % Parameters.checkpoint_mod == 0:
                    cv.imwrite('{}/checkpoint.jpg'.format(Parameters.dir_out, i), generation[0].genes)

                if i % Parameters.save_mod == 0:
                    cv.imwrite('{}/cycle{}.jpg'.format(Parameters.dir_out, i), generation[0].genes)

                print('Best Score: {}, similarity: {}%'
                      .format(generation[0].score,
                              100.0*(max_score - generation[0].score)/max_score))

            else:
                children = Genetics.crossover(generation)
                np.concatenate((generation, children), out=generation)
                Genetics.np_mutation(generation)
                generation.sort()
                generation = Genetics.selection(generation)

                if len(generation) > Parameters.population_size:
                    generation = generation[:Parameters.population_size]

                if generation[0].score < best_score:
                    best_score = generation[0].score
                    best_picture = generation[0].genes

                if i % Parameters.checkpoint_mod == 0:
                    cv.imwrite('{}/checkpoint.jpg'.format(Parameters.dir_out, i),
                               generation[0].genes)

        return best_picture, best_score


best_picture, best_score = Utils.track_time(Genetics.evolution, name='All in all')

cv.imwrite('{}/result.jpg'.format(Parameters.dir_out), best_picture)
print('Best score is:', best_score)
