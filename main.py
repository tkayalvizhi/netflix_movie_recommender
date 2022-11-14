import numpy as np
import kmeans
import common
import naive_em
import pandas as pd
import em

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

# TODO: Your code here

for K in [1, 2, 3 , 4, 5]:
    print(f'K: {K}')
    # kmeans_df = pd.DataFrame(columns=['mixture', 'post', 'cost'])
    em_df = pd.DataFrame(columns=['mixture', 'post', 'cost'])
    for seed in [0, 1, 2, 3, 4]:
        mixture, post = common.init(X, K, seed)
        # kmeans_df.loc[seed] = kmeans.run(X, mixture, post)
        em_df.loc[seed] = em.run(X, mixture, post)

    # print(kmeans_df['cost'].min())
    print(f"{K}: {em_df['cost'].max()}")

    # m_km, p_km, c_km = kmeans_df.loc[kmeans_df['cost'].argmin()]
    m_em, p_em, c_em = em_df.loc[em_df['cost'].argmax()]
    X_pred = em.fill_matrix(X, m_em)
    print(common.rmse(X_gold, X_pred))

    # common.plot(X, m_km, p_km, f'K = {K}, Kmeans algorithm')
    # common.plot(X, m_em, p_em, f'K = {K}, EM algorithm')
    #
    # print(f'BIC for K = {K}: {common.bic(X, m_em, c_em)}')