#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Wen Jiang <phdcluster@gmail.com>"

# matplotlib 	2.1.1
# pandas 		0.22.0
# scipy		    1.0.0

import math
import scipy.stats
import matplotlib.pylab as plt
import matplotlib.ticker as ticker

#𝑓(𝜎)represents the Black Scholes Call Option Price
def function(s, E, r, t, c, sigma):

    d1 = (math.log(s / E) + (r + sigma ** 2 / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    #With cumulative normal distribution function
    return s * scipy.stats.norm.cdf(d1) - E * math.exp(r * t) * scipy.stats.norm.cdf(d2) - c

#𝑓′(𝜎) represents the derivative of 𝑓(𝜎)
def differentiated(s, E, r, t, c, sigma):

    d1 = (math.log(s / E) + (r + sigma ** 2 / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    # Derivative of d1 w.r.t.sigma
    dd1 = (sigma ** 2 * t * math.sqrt(t) - (math.log(s / E) + (r + sigma ** 2 / 2) * t) * math.sqrt(t)) / (sigma ** 2 * t)
    # Derivative of d2 w.r.t.sigma
    dd2 = dd1 - math.sqrt(t)

    #With normal probability density function
    return s * scipy.stats.norm.pdf(d1) * dd1 - E * math.exp(-r * t) * scipy.stats.norm.pdf(d2) * dd2

#Newton-Raphson Algorithm for calculating implied volatility:
def newtonRaphson(s, E, r, t, c):

    #parameters according to the slides
    sigma = 0.1
    tolerance = 0.00000001

    answer = [0]*10
    answer[1]=sigma

    maxIteration = 100
    i = 2

    while (i <= maxIteration):
        # 𝑓(𝑥0)/𝑓′(𝑥0)
        fun = function(s, E, r, t, c, sigma)
        dif = differentiated(s, E, r, t, c, sigma)

        #Set 𝑥 = 𝑥0 − 𝑓(𝑥0)/𝑓′(𝑥0)
        sigma = sigma - fun / dif
        answer[i]= sigma

        #STOP when |𝜎𝑛+1 − 𝜎𝑛| is small
        if (abs(answer[i] - answer[i - 1]) < tolerance):
            answer=answer[1:i]
            break
        i=i+1

    return answer

def main():

    #Lecture slide
    #s = 2.36
    #E = 2.36
    #r = 0.01
    #t = 1
    #c = 0.1875
    #[0.1, 0.21406361336895435, 0.21175265608018426, 0.21178641062518216, 0.2117859176558003]

    #Project slide
    s = 34 #(Yahoo current stock price)
    E = 34 #(excise price)
    r = 0.001 #(risk free rate)
    t = 1 #( time to expiry)
    c = 2.7240 #(call option price)
    #[0.1, 0.20226965555865664, 0.20233009047936534, 0.2023300271196214]

    solution = (newtonRaphson(s, E, r, t, c))

    # Friendly reminder on closing the graphical windows apart from the command-line interface if needed
    print("Close the plot windows if needed to continue")

    correctIndex = (range(1, len(solution)+1,1))

    # matplotlib plotting
    fig, ax = plt.subplots()
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.plot(correctIndex, solution, color='red', label="value")
    plt.xlabel('Number of Estimate')
    plt.ylabel('Estimate')
    plt.legend(loc='lower center')
    plt.title('Calculation of Implied Volatility Using The Newton Raphson Algorithm')

    # always display the best approximation
    for i, j in zip(correctIndex, solution):
        if(i==len(correctIndex)):
            ax.annotate('%s' % round(j,4), xy=(i, j), xytext=(0, 0), textcoords='offset points')

    plt.show()

# main of our program to prevent python script execution upon call
if __name__ == "__main__":
    main()
