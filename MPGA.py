#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Wen Jiang"

"""
To follow the requirements of the class, I used the WQU started code listed in the algorithm class to get a jump on the first two steps as told from Douglas Kelly.
I read the Meyer and Packard online in details myself mainly for this class. for example http://www.mty.itesm.mx/dtie/centros/csi/materias/ia-5005/apuntes/mitchell-prediccion.pdf

Genetic algorithms (GA) optimize using select, crossover, and mutate operations.

The steps to this Mini-project, which is working with the Packard Maynard Generic Algorithm, are:
   1 - Initialize the population with a random set of Cs
   2 - Calculate the fitness of each C
   3 - Rank the population by fitness
   4 - Discard lower-fitness individuals and replace with new Cs obtained by
       applying crossover and mutation to the remaining Cs
   5 - Repeat from step 2
   6 - Once highest fit C's obtained - predict next 24 months of stock prices
   7 - Then improve the algorithm to get even better predication's
   
For relevant details, please see https://piazza.com/class/je0aheja9jo1od?cid=17
"""

# matplotlib 	2.1.1
# scipy		    1.0.0
# numpy		    1.4.0
# pandas 		0.22.0

import quandl
import random
import math
import numpy as np
import matplotlib.pyplot as plt


def get_data(symbol="WIKI/TSLA", start="2013-10-08", end="2017-10-08", field="Adj. Close"):
    alldata = quandl.get(symbol, authtoken='MgUeDFrhs9JjaA4gKzEx')  # Must use your own authentication token thereafer
    mydata = alldata.loc[start:end, field]

    return (mydata)


def generate_individual(min_price, max_price, n_tuples):
    if (max_price <= min_price) or (n_tuples < 2) or (n_tuples > 10):
        print("Invalid parameters (min_price = %d, max_price=%d, n_tuples = %d" % (min_price, max_price, n_tuples))
        return

    # Compute the increment (or range) of prices
    inc = int(round((max_price - min_price) / (n_tuples + 1)))

    # Assign the initial price range between start and next
    start_price = min_price
    next_price = min_price + inc

    # Randomaly assign first conditional prices
    a = random.randint(start_price, next_price)
    b = random.randint(next_price + 1, next_price + (2 * inc))

    # Check how many tuples requested
    if n_tuples == 2:  # Only 2 so done
        individuals = (a, b)
    elif n_tuples == 3:  # 3, so compute third conditional price
        c = random.randint(next_price + (2 * inc) + 1, max_price)
        individuals = (a, b, c)
    elif n_tuples == 4:  # 4, so compute all four conditional prices
        c = random.randint(next_price + (2 * inc) + 1, next_price + (3 * inc))
        d = random.randint(next_price + (3 * inc) + 1, max_price)
        individuals = (a, b, c, d)
    elif n_tuples == 5:  # 5, so compute all five conditional prices
        c = random.randint(next_price + (2 * inc) + 1, next_price + (3 * inc))
        d = random.randint(next_price + (3 * inc) + 1, next_price + (4 * inc))
        e = random.randint(next_price + (4 * inc) + 1, max_price)
        individuals = (a, b, c, d, e)
    elif n_tuples == 6:  # 6 so compute all six conditional prices
        c = random.randint(next_price + (2 * inc) + 1, next_price + (3 * inc))
        d = random.randint(next_price + (3 * inc) + 1, next_price + (4 * inc))
        e = random.randint(next_price + (4 * inc) + 1, next_price + (5 * inc))
        f = random.randint(next_price + (5 * inc) + 1, max_price)
        individuals = (a, b, c, d, e, f)
    elif n_tuples == 7:  # 7 so compute all six conditional prices
        c = random.randint(next_price + (2 * inc) + 1, next_price + (3 * inc))
        d = random.randint(next_price + (3 * inc) + 1, next_price + (4 * inc))
        e = random.randint(next_price + (4 * inc) + 1, next_price + (5 * inc))
        f = random.randint(next_price + (5 * inc) + 1, next_price + (6 * inc))
        g = random.randint(next_price + (6 * inc) + 1, max_price)
        individuals = (a, b, c, d, e, f, g)
    elif n_tuples == 8:  # 8 so compute all six conditional prices
        c = random.randint(next_price + (2 * inc) + 1, next_price + (3 * inc))
        d = random.randint(next_price + (3 * inc) + 1, next_price + (4 * inc))
        e = random.randint(next_price + (4 * inc) + 1, next_price + (5 * inc))
        f = random.randint(next_price + (5 * inc) + 1, next_price + (6 * inc))
        g = random.randint(next_price + (6 * inc) + 1, next_price + (7 * inc))
        h = random.randint(next_price + (7 * inc) + 1, max_price)
        individuals = (a, b, c, d, e, f, g, h)
    elif n_tuples == 9:  # 9 so compute all six conditional prices
        c = random.randint(next_price + (2 * inc) + 1, next_price + (3 * inc))
        d = random.randint(next_price + (3 * inc) + 1, next_price + (4 * inc))
        e = random.randint(next_price + (4 * inc) + 1, next_price + (5 * inc))
        f = random.randint(next_price + (5 * inc) + 1, next_price + (6 * inc))
        g = random.randint(next_price + (6 * inc) + 1, next_price + (7 * inc))
        h = random.randint(next_price + (7 * inc) + 1, next_price + (8 * inc))
        i = random.randint(next_price + (8 * inc) + 1, max_price)
        individuals = (a, b, c, d, e, f, g, h, i)
    else:  # 10 (or more), so compute 10 conditinal prices
        c = random.randint(next_price + (2 * inc) + 1, next_price + (3 * inc))
        d = random.randint(next_price + (3 * inc) + 1, next_price + (4 * inc))
        e = random.randint(next_price + (4 * inc) + 1, next_price + (5 * inc))
        f = random.randint(next_price + (5 * inc) + 1, next_price + (6 * inc))
        g = random.randint(next_price + (6 * inc) + 1, next_price + (7 * inc))
        h = random.randint(next_price + (7 * inc) + 1, next_price + (8 * inc))
        i = random.randint(next_price + (8 * inc) + 1, next_price + (9 * inc))
        j = random.randint(next_price + (9 * inc) + 1, max_price)
        individuals = (a, b, c, d, e, f, g, h, i, j)

    # Return set of individual conditional prices
    return individuals


##################################################################
# Fuction is condition_satisfied
#
#   Inputs:
#     a = current index of allele
#     allele = individual set of conditions
#     s = current index of stock data
#     stock = time series data of stock prices
#     n_tuples = how many conditions per tuples stock prices are to satisfy
#
#   Output:
#     True - if condition met
#     False - otherwise
#
# These can be ANY conditions or rules you choose to define
##################################################################
def condition_satisfied(a, allele, s, stock, n_tuples):
    # The is ONLY valild for allels of 3, 4, 5, and 6.

    if n_tuples == 3:
        # Three conditions C1, C2, C3 such that C1 <= P1 < C2 <= P2 < C3
        if stock[s - 1] >= allele[a][0] and \
                        stock[s] >= allele[a][1] and \
                        stock[s + 1] <= allele[a][2]:
            return True

    if n_tuples == 4:
        # Four conditions C1, C2, C3, C4 such that C1 >= P1 < C2 <= P2 < C3 <= P3 <= C4
        if (stock[s - 1] >= allele[a][0] or stock[s] >= allele[a][1]) and \
                (stock[s] >= allele[a][1] or stock[s + 1] <= allele[a][3]):
            return True

    if n_tuples == 5:
        # Five conditions C1, C2, C3, C4, C5 such that P1 >= C1, P2 >= C2 or <= C3, and P3 >= C4 or <= C5
        if stock[s - 1] >= allele[a][0] and \
                (stock[s] >= allele[a][1] or stock[s] <= allele[a][2]) and \
                (stock[s + 1] >= allele[a][3] or stock[s + 1] <= allele[a][4]):
            return True

    if n_tuples == 6:
        # Six conditions C1, C2, C3, C4, C5, C6 such that P1 >= C1, P2 >= C2 or <= C3, and P3 >= C4 or <= C5
        if (stock[s - 1] >= allele[a][0] and stock[s - 1] < allele[a][1]) and \
                (stock[s] >= allele[a][1] and stock[s] < allele[a][2]) and \
                (stock[s + 1] >= allele[a][2] and stock[s + 1] < allele[a][3]) and \
                (stock[s + 2] >= allele[a][3] and stock[s + 2] < allele[a][4]) and \
                (stock[s + 3] >= allele[a][4] and stock[s + 3] <= allele[a][5]):
            return True
            # if n_tuples == 7:
            # if n_tuples == 8:
            # if n_tuples == 9:
            # if n_tuples == 10:


##################################################################
# Step 1 - Initialize the population with a random set of C
#
# Function is generate_population(size, min_price, max_price, n_tuples)
#
#  Inputs:
#     size = the population size to generate, default is 100
#     min_price = lowest price condition, default is 141 (Telsa price)
#     max_price = higher price condition, default is 390 (Telsa price)
#     n_tuples = number of tuples (5 or fewer)
#
#   Output:
#     C = random set of indiviudals in the population
#
##################################################################
def generate_population(size, min_price, max_price, n_tuples):
    C = []

    # Loop to create entire population of C's
    for i in range(0, size):
        # Append individual conditions
        C.append(generate_individual(min_price, max_price, n_tuples))
        # Return the random set of conditions in entire population
    return C


##########################################################################
# collect stock Y-values that pass the series of conditions

def findYs(individuals, timeSeries, n_tuples):
    myList = []
    myDict = {}

    # Look at each individual C in the population and see if a fix exists
    for y in range(len(individuals)):
        # Examine the adjusted closing price to see if condition hold for the Y-values
        for x in range(1, len(timeSeries) - 1):

            if condition_satisfied(y, individuals, x, timeSeries, n_tuples):
                myList.append(timeSeries[x])
                myDict[individuals[y]] = myList

        # Reset list for next individual condition C
        myList = []

    # Return all conditions subsequent Y-values which have been satisfied
    return myDict


##########################################################################

##########################################################################
def popfitness(population, Yvals, std, std_o):
    """Determine the fitness of every individual in the population."""
    popfit = []
    for i in range(0, len(population)):
        popfit.append(fitness(population[i], Yvals, std, std_o))
    return popfit


##########################################################################
#      Fitness Function:
#        f(C) = -log2(std/std_o) - alpha/N_c
#        where:
#        std: standard deviation of the x set that satisfy condition C
#        std_o: the standard deviation of the distribution of x over the entire dataset
#        N_c: the number of data points satisfying condition C
#        alpha: Constant
##########################################################################
def fitness(individual, Yvals, std, std_o, alpha=1):
    myFitness = []

    if len(Yvals) > 1:
        myFitness.append(-math.log(std / std_o - (alpha / len(Yvals))))  # Fitness function
    else:
        myFitness.append(0)
    return myFitness


def fitfun(Yvals, mydata):
    std_o = np.std(mydata)
    i = 0
    popfitlist = []
    conlist = []
    genpopulationlist = []

    for k in Yvals.keys():
        popfit = popfitness(individuals, Yvals, np.std(Yvals[k]), std_o)

        genpopulationlist.append(list(Yvals[k]))

        print("For the Condition")
        print(k)

        conlist.append(k)
        popfitlist.append(popfit[++i][0])
        print("fitness value = %s" % popfit[++i][0])

    return popfitlist, conlist, genpopulationlist


def computeFitness(individuals, mydata, number_of_tuples):
    # Find those Y-valus which match the conditions C
    Yvals = findYs(individuals, mydata, number_of_tuples)

    # Display fitness values of satisfied conditions C
    print("**********************************************************")
    print("Step 2: Fitness Values")
    print("**********************************************************")

    popfitlist, conlist, genpopulationlist = fitfun(Yvals, mydata)

    return Yvals, popfitlist, conlist, genpopulationlist


def rankfit(popfitlist):
    print("**********************************************************")
    print("Step 3 - Rank Fitness by Population")
    print("**********************************************************")
    # Higher the fitness the better.

    minIndex = popfitlist.index(min(popfitlist))
    print("Minimum Fitness Index\t\t\t", minIndex)
    print("Minimum Fitness Condition\t\t", conlist[minIndex])

    maxIndex = popfitlist.index(max(popfitlist))
    print("Maximum Fitness Index\t\t\t", maxIndex)
    print("Maximum Fitness Condition\t\t", conlist[maxIndex])

    sortedlist = sorted(popfitlist)
    print("Ranked Fitness by Population\t", sortedlist)

    return minIndex, maxIndex, sortedlist


##########################################################################
#       Select best fit individuals for reproduction
##########################################################################
def select(genpopulationlist, minIndex):
    if (len(genpopulationlist) == 4):
        samelength = min(
            len(i) for i in [genpopulationlist[0], genpopulationlist[1], genpopulationlist[2], genpopulationlist[3]])
    elif (len(genpopulationlist) == 3):
        samelength = min(
            len(i) for i in [genpopulationlist[0], genpopulationlist[1], genpopulationlist[2]])
    elif (len(genpopulationlist) == 2):
        samelength = min(
            len(i) for i in [genpopulationlist[0], genpopulationlist[1]])
    elif (len(genpopulationlist) == 1):
        samelength = min(
            len(i) for i in [genpopulationlist[0]])

    if (len(genpopulationlist) != 1):
        del genpopulationlist[minIndex]

        i = 0
        while (i < len(genpopulationlist)):
            if (len(genpopulationlist[i]) > samelength):
                del genpopulationlist[i][-(len(genpopulationlist[i]) - samelength):]
            i = i + 1

    return genpopulationlist


def crossover(result):
    uniform_crossover_rate = .5
    crossedover = []

    k = 0

    # uniform crossover
    while (k < len(result) - 1):

        # tuple can't be crossed over
        if (k == 0):
            AL = list(result[k])
        else:
            AL = BL

        BL = list(result[k + 1])

        for i in range(len(result[k])):
            if uniform_crossover_rate > random.random():
                # apply crossover
                temp = AL[i]
                AL[i] = BL[i]
                BL[i] = temp

        crossedover.append(AL)

        if (k == len(result) - 2):
            crossedover.append(BL)

        k = k + 1

    return crossedover


def mutation(offspring):
    chance_to_mutate = .01
    i = 0
    for p in offspring:
        while (i < len(p)):
            if chance_to_mutate > random.random():
                spot = random.randint(0, len(p) - 1)
                p[spot] = float("{0:.2f}".format(random.uniform(min(p), max(p))))

            i = i + 1
        i = 0

    return offspring


def selcrossmut(genpopulationlist, minIndex):
    print("**********************************************************")
    print("Step 4 - Discard and replace C's using select, mutate, and crossover operations")
    print("**********************************************************")

    conlistResult = select(genpopulationlist, minIndex)
    for p in conlistResult:
        print("C's after selection\t\t\t\t", p)

    offspring = crossover(conlistResult)
    for p in offspring:
        print("C's after crossover\t\t\t\t", p)

    newoffspring = mutation(offspring)
    for p in newoffspring:
        print("C's after mutation\t\t\t\t", p)

    return conlistResult, offspring, newoffspring


def roll(data):
    short_rolling = data.rolling(window=20).mean()
    long_rolling = data.rolling(window=50).mean()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data.index, data, label=str(len(data)) + ' Datapoints of stock prices')
    ax.plot(short_rolling.index, short_rolling, label='20 days rolling')
    ax.plot(long_rolling.index, long_rolling, label='50 days rolling')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.legend()
    plt.show()


if __name__ == '__main__':

    ##########################################################################
    # Main program -- load data, generate chromosones, and test fitness
    #
    # This implements the Meyer Packard Genetic Algorithm in 6 steps:
    #   1 - Initialize the population with a random set of C
    #   2 - Calculate the fitness of each C
    #   3 - Rank the population by fitness
    #   4 - Discard lower-fitness individuals and replace with new Cs obtained by
    #       applying crossover and mutation to the remaining Cs
    #   5 - Repeat from step 2
    #   6 - Once highest fit C's obtained - predict next 24 months of stock prices
    #   7 - Improve the algorithms to get better predication's
    #
    # This is SAMPLE or STARTER code only.  There are many ways to approach
    # this problem.  Be creative.  Only Steps 1 and 2 above have been coded.
    # Also, the code has not been fully tested -- so use with care and modify as needed
    ##########################################################################


    quandl_symbol = "WIKI/TSLA"  # Replace "AAL" with any stock, "TSLA" is the default
    mydata = get_data(quandl_symbol)  # Load time series data from Quandl

    # Use the minimum and maximum prices as the low and high prices (but these can be set to different values)
    lowest_price = int(min(mydata))  # Lowest price to test price conditions
    highest_price = int(max(mydata))  # Highest price to test price conditions

    # lowest_price = 15         # Lowest price to test price conditions
    # highest_price = 20       # Highest price to test price conditions

    # Set the number of C's the series must satisfy and how many to generate in the population
    number_of_tuples = 5
    number_of_individuals = 4

    ################################################################
    # 1 - Generate the population of random individuals of C
    ################################################################
    individuals = generate_population(number_of_individuals, lowest_price, highest_price, number_of_tuples)
    print("individuals ", individuals)
    print("**********************************************************")
    print("Step 1: The randomly generated population of C's are:")
    print("**********************************************************")

    ################################################################
    # 2 - Compute the fitness of each C
    ################################################################

    Yvals, popfitlist, conlist, genpopulationlist = computeFitness(individuals, mydata, number_of_tuples)

    ################################################################
    # 3 - Rank Fitness by Population
    ################################################################

    minIndex, maxIndex, sortedlist = rankfit(popfitlist)

    ################################################################
    # 4 - Discard and replace C's using select, mutate, and crossover operations
    ################################################################

    conlistResult, offspring, newoffspring = selcrossmut(genpopulationlist, minIndex)

    ################################################################
    # 5 - Repeat Step #2 until confident "best" conditions identified
    ################################################################

    # use the result from select, mutate, and crossover operations
    # ie select the best fit individuals for reproduction
    counter = 0
    while (len(newoffspring) > 1):
        newindividuals = []
        for p in newoffspring:
            min_price = int(min(p))
            max_price = int(max(p))

            individ = generate_individual(min_price, max_price, 5)
            newindividuals.append(individ)
        print(newindividuals)

        ################################################################
        # 2 - Compute the fitness of each C
        ################################################################

        Yvals, popfitlist, conlist, genpopulationlist = computeFitness(newindividuals, mydata, number_of_tuples)

        ################################################################
        # 3 - Rank Fitness by Population
        ################################################################

        minIndex, maxIndex, sortedlist = rankfit(popfitlist)

        ################################################################
        # 4 - Discard and replace C's using select, mutate, and crossover operations
        ################################################################

        conlistResult, offspring, newoffspring = selcrossmut(genpopulationlist, minIndex)

    ################################################################
    # 6 - Now, use those conditions to predict next 24 months on prices (save to csv file)
    ################################################################

    print("Fitness by Population", sortedlist[1])

    # best solution in current population
    for p in conlistResult:
        print("highest fit C's", p)
        plt.plot(p)
        plt.title(str(len(p)) + " prices from the best chromosome")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.show()

    print("Fitness by Population", sortedlist[1])

    # synchronize the indicators
    short_rolling = mydata.rolling(window=10).mean()
    chrom_rolling = p
    long_rolling = mydata.rolling(window=30).mean()

    mydata = mydata[(len(mydata) - len(chrom_rolling)):]
    short_rolling = short_rolling[(len(long_rolling) - len(chrom_rolling)):]
    long_rolling = long_rolling[(len(long_rolling) - len(chrom_rolling)):]

    # compare the performance of the best chromosome with rolling windows
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(mydata.index, mydata, label='datapoints of stock prices')
    ax.plot(short_rolling.index, p, label='datapoints of chromosome prices')
    ax.plot(short_rolling.index, short_rolling, label='10 days rolling')
    ax.plot(long_rolling.index, long_rolling, label='30 days rolling')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    ax.legend()
    plt.show()

