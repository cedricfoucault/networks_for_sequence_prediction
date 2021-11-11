# TransitionProbModel

Python version of the (matlab) MinimimalTransitionProbsModel.  
https://github.com/florentmeyniel/MinimalTransitionProbsModel  
This toolbox computes posterior inferences from a sequence of observations.  
The inference can be:
- Bayes optimal solution assuming no change point
    - with perfect memory
    - with an exponential leak on observations count
    - within a sliding window of observations
- Bayes optimal solution assuming change point (computed with Hidden Markov Model and numeric integration).

The Python version is more general:
- it is not restricted to the binary case: it can handle an arbitrary number of item
- it is not restricted to order-0 and order-1 transition probabilities: it can handle an arbitrary order

NB: due to numerical overflow, the inference will crash for higher order and large number of items.
