import pandas as pd

"""
Weighted Rating - 

(v/v+m)*R + (m/v+m)*C

v is the number of votes (vote_count)
m is the minimum votes required to be listed in chart
R is the average rating for the movie (vote_average)
C is the mean vote across the whole report
"""
data = pd.read_csv("movies_metadata.csv")

print (data.info())
print(data.head())

C = data["vote_average"].mean()
print(C)

M = data["vote_count"].quantile(0.90)
print(M)

q_movie = data.copy().loc[data["vote_count"]>=M]
print(q_movie.shape)

def W_R(x, M=M, C=C):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v/v+M)*R + (M/v+M)*C

q_movie["score"] = q_movie.apply(W_R, axis = 1)
print(q_movie.head())

q_movie = q_movie.sort_values("score", ascending = False)
print(q_movie.head())

print(q_movie[["title", "vote_count", "vote_average", "score"]].head(10))