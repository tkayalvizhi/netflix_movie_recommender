# netflix_movie_recommender
This project was done as part of the Machine Learning with Python-From Linear Models to Deep Learning course [MITx MicroMasters in Statistics and Data Science]( https://micromasters.mit.edu/ds/)

The task is to build a mixture model for collaborative filtering. We are given a data matrix containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled. The goal is to predict all the remaining entries of the matrix.

We will use mixtures of Gaussians to solve this problem. The model assumes that each user's rating profile is a sample from a mixture model. In other words, we have K possible types of users and, in the context of each user, we must sample a user type and then the rating profile from the Gaussian distribution associated with the type. We will use the Expectation Maximization (EM) algorithm to estimate such a mixture from a partially observed rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step). Once we have the mixture, we can use it to predict values for all the missing entries in the data matrix.

## Functions implemented by me

1. [naive_em.py](https://github.com/tkayalvizhi/netflix_movie_recommender/blob/c1ec10da61e2ff326382e5d4204e29665662616d/naive_em.py) - a first version of the EM algorithm 
2. [em.py](https://github.com/tkayalvizhi/netflix_movie_recommender/blob/c1ec10da61e2ff326382e5d4204e29665662616d/em.py) - a mixture model for collaborative filtering 
3. [common.py](https://github.com/tkayalvizhi/netflix_movie_recommender/blob/c1ec10da61e2ff326382e5d4204e29665662616d/common.py) - the common functions for all models 
4. [main.py](https://github.com/tkayalvizhi/netflix_movie_recommender/blob/c1ec10da61e2ff326382e5d4204e29665662616d/main.py) - code to answer the questions for this project
5. [test.py](https://github.com/tkayalvizhi/netflix_movie_recommender/blob/c1ec10da61e2ff326382e5d4204e29665662616d/test.py) - code to test the implementation of EM for a given test case
