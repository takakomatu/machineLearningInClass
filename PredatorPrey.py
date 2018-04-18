import matplotlib.pyplot as plt

# K, r, s, u, and v are parameters of the model.
# p and q are initial populations.
# n is the number of generations.

def predator_prey(K, r, s, u, v, p, q, n): 
    pop1 = [p]
    pop2 = [q]
    for i in range(n):
        last1 = pop1[-1]
        last2 = pop2[-1]
        
        new1 = last1 * (1 + r * (1 - last1 / K)) - (s * last1 * last2) # Lakota-
        new2 = (1 - u) * last2 + v * last1 * last2 #zombies # new2 is similar to the model equation on the poster
        
        pop1.append(new1)
        pop2.append(new2)
        
    plt.close()
    plt.plot(pop1, 'o--')
    plt.plot(pop2, 'o--')
    plt.ylabel("Number of humans/zombies")
    
    plt.show()
    
predator_prey(.8, -0.6, 0.5, 0.7, 1.6, 0.4, 0.05, 25)
# p for number of humans, q for number of zomibies, n for maximam number of x axis
# 

#https://www.ssc.wisc.edu/~jmontgom/predatorprey.pdf
#https://tuhat.helsinki.fi/portal/files/10254948/ComplexDynamicBehaviorsOfADiscreteTimePredatorPreySystem.pdf
# look pdf above but dont know which fomula you use.

