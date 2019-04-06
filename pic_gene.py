#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
from skimage.measure import compare_ssim
import cv2 as cv

# pic = cv.imread('star_sky.jpeg', flags=cv.IMREAD_COLOR)
pic = cv.imread('puzzle.jpg', flags=cv.IMREAD_COLOR)
# ## Parameters

# In[16]:


population_size = 1000
generations_size = 1000
initial_generation_limit = 15

# Figures
color_transparency = 100
## Circle
min_radius = 4
max_radius = 10
## Polygons
min_edge_length = 10
max_edge_length = 100

# Mutation
mutation_chance = 0.4
min_mutation_amount = 5
max_mutation_amount = 20

# Crossover
min_crossover_amount = 10
max_crossover_amount = 100

# Selection
selection_percent = 0.2


# ## Figure classes

# In[6]:


class Circle:
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

    def draw(self, picture):
        cv.circle(picture, self.center, self.radius, self.color, thickness=cv.FILLED)

    @staticmethod
    def get_random():
        radius = np.random.randint(low=min_radius, high=max_radius)

        x = np.random.randint(0, pic.shape[0])
        y = np.random.randint(0, pic.shape[1])
        color = (np.random.randint(256), 0, 0, color_transparency)

        return Circle((x, y), radius, color)


class Polygon:
    def __init__(self, points, color):
        self.points = points
        self.color = color

    def draw(self, picture):
        points = np.array(self.points, dtype=np.int32)
        # points = points.reshape((-1, 1, 2))
        cv.fillPoly(picture, [points], self.color)

    @staticmethod
    def get_random(n_vertices=3):
        points = []
        curr_point = (np.random.randint(0, pic.shape[0]), np.random.randint(0, pic.shape[1]))
        points.append(curr_point)
        for _ in range(n_vertices - 1):
            new_point = [np.random.randint(curr_point[0], curr_point[0] + max_edge_length),
                         np.random.randint(curr_point[1], curr_point[1] + max_edge_length)]

            # new_point = [np.random.randint(0, pic.shape[0]), np.random.randint(0, pic.shape[1])]
            curr_point = new_point
            points.append(curr_point)

        color = (np.random.randint(256), np.random.randint(256),
                 np.random.randint(256), color_transparency)
        return Polygon(points, color)


# ## Chromosome class 

# In[9]:

class Chromosome:
    def __init__(self, genes=None):
        if genes is None:
            genes = []

        self.rendered_figures = []
        self.to_render = genes
        self.picture = np.zeros_like(pic)
        self.score = self.__fitness_score(self.picture)

    def mutate(self):
        choice = np.random.uniform()
        if choice >= mutation_chance:
            return
        #
        # min_mutation = int(min_mutation_amount*self.score)
        # max_mutation = int(max_mutation_amount*self.score)
        min_mutation = min_mutation_amount
        max_mutation = max_mutation_amount
        mutation_amount = np.random.randint(min_mutation, max_mutation)

        for _ in range(mutation_amount):
            figure_choice = np.random.randint(2)

            #TODO This is temporary
            figure_choice = 1

            figure = None
            if figure_choice == 0:
                figure = Polygon.get_random()

            if figure_choice == 1:
                figure = Circle.get_random()

            self.to_render.append(figure)

    def update_score(self, original=pic):
        start = timer()
        self.render()
        end = timer()
        # print("Render elapsed", str(end - start))
        start = timer()
        self.score = self.__fitness_score(self.picture, original)
        end = timer()
        # print("Fitness score elapsed", str(end - start))

    def render(self):
        if not self.to_render:
            return
        for figure in self.to_render:
            figure.draw(self.picture)

        self.rendered_figures += self.to_render
        self.to_render = []

    @staticmethod
    def __fitness_score(current, original=pic):

        curr_grey = cv.cvtColor(current, cv.COLOR_BGR2GRAY)
        orig_gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

        res = np.square(np.subtract(curr_grey, orig_gray))
        return res.sum()/(current.shape[0]*current.shape[1])
        # res = np.subtract(current, original)
        # res = np.square(res)
        # return np.sum(res)/(current.shape[0]*current.shape[1])

    @staticmethod
    def cross2(sample1, sample2):
        genes1_len = len(sample1.to_render) + len(sample1.rendered_figures)
        genes2_len = len(sample2.to_render) + len(sample2.rendered_figures)

        if genes1_len == 0:
            return sample2

        if genes2_len == 0:
            return sample1

        gene_pull = np.concatenate((sample1.to_render, sample1.rendered_figures,
                                    sample2.to_render, sample2.rendered_figures))

        outcomes1 = sample1.score*genes1_len
        outcomes2 = sample2.score*genes2_len
        all_outcomes = outcomes1 + outcomes2
        p1 = outcomes1/all_outcomes
        p2 = outcomes2/all_outcomes

        if p1 == p2:
            p = np.ones(all_outcomes)*p1
        else:
            p1_arr = np.ones(genes1_len)*p1
            p2_arr = np.ones(genes2_len)*p2

            p = np.concatenate(p1_arr, p2_arr)

        # p = np.array([p1/genes1_len]*genes1_len + [p2/genes2_len]*genes2_len)
        # print('Probs:', p)
        cross_amount = (genes1_len + genes2_len)//2
        child = np.random.choice(gene_pull, size=cross_amount, replace=False, p=p)

        return Chromosome(list(child))

        # genes1_len = len(sample1.rendered_figures) + len(sample1.to_render)
        # genes2_len = len(sample2.rendered_figures) + len(sample2.to_render)
        #
        # outcomes1 = sample1.score*genes1_len
        # outcomes2 = sample2.score*genes2_len
        # all_outcomes = outcomes1 + outcomes2
        #
        # p1 = outcomes1/all_outcomes
        # p2 = outcomes2/all_outcomes
        # res = cv.addWeighted(sample1.picture, p1, sample2.picture, p2, 0)
        # new_chromosome = Chromosome()
        # new_chromosome.picture = res
        # return new_chromosome

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return str(self.score)


np_mutation = np.vectorize(Chromosome.mutate)
np_update_score = np.vectorize(Chromosome.update_score)
np_render = np.vectorize(Chromosome.render)

# ## Utilities for the genetic algorithm

# In[11]:


def crossover(generation):
    cross_number = np.random.randint(min_crossover_amount, max_crossover_amount)
    cross_number = min(cross_number, len(generation))

    res = []
    for i in range(1, cross_number):
        child = Chromosome.cross2(generation[i - 1], generation[i])
        res.append(child)

    return res


def selection(generation):
    n_remove = int(selection_percent*len(generation))
    return generation[: len(generation) - n_remove]


# In[105]:

from timeit import default_timer as timer


def track_time(func, name=None, *args):
    if name is None:
        name = func.__name__

    start = timer()
    result = func(*args)
    end = timer()

    print(name, 'elapsed', str(end - start), 'seconds')

    return result


def evolution():
    generation = np.array([Chromosome() for _ in range(initial_generation_limit)])
    best_picture = generation[0].picture
    best_score = generation[0].score

    for i in range(generations_size):

        # print("\n------------------------\nIteration {}\n".format(i))
        # children = track_time(crossover, 'Crossover', generation)
        # generation = np.concatenate((generation, children))
        #
        # track_time(np_mutation, 'Mutation', generation)
        # track_time(np_render, 'Render', generation)
        # track_time(np_update_score, 'Score Calculation', generation)
        # track_time(np.sort, 'Sort', generation)
        #
        # generation = track_time(selection, 'Selection', generation)

        children = crossover(generation)
        generation = np.concatenate((generation, children))
        np_mutation(generation)
        np_render(generation)
        np_update_score(generation)
        generation = np.sort(generation)
        generation = selection(generation)

        if generation[0].score < best_score:
            best_score = generation[0].score
            best_picture = generation[0].picture

        # while len(generation) > population_size:
        #     generation.pop()

        cv.imwrite('progress/cycle{}.img{}.jpg'.format(i, 0), generation[0].picture)

    return best_picture, best_score


best_picture, best_score = track_time(evolution, name='All in all')

cv.imwrite('result.png', best_picture)
print('Best score is:', best_score)

# In[106]

#
# generation = np.array([Chromosome() for _ in range(initial_generation_limit)])
#
# np_update_score = np.vectorize(Chromosome.update_score)
# np_mutate = np.vectorize(Chromosome.mutate)
#
# start = timer()
# for _ in range(100):
#     np_mutate(generation)
#     np_update_score(generation)
# end = timer()
#
# print('Time elapsed:', str(end - start))
# cv.imwrite('result.png', generation[0].picture)
