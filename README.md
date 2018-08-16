# Movies Recommendation System

## Data Brief Description
The data is given as 3 'txt' files:
  1. Ratings  -  UserID::MovieID::Rating::Timestamp
  2. Users    -  UserID::Gender::Age::Occupation::Zip-code
  3. Movies   -  MovieID::Title::Genres
 
The full data description can be found in the file data/README

## Project Description
Purpose: Recommend a user on the best movies for him, based on his movies' rating history and other users preferences.

The project include Matrix Factorization (MF) models (see [1]), and two possible approches to minimize the cost function - Stochastic Gradient Descent (SGD) and Alternating Least Squares(ALS)

## Evaluation
Evaluation metrics in order to assess the performance: (implemented in 'metrics')

1. Root Mean Square Error (RMSE)
2. Mean Percentile Rank (MPR)
3. Precision at K (P@K)
4. Recall at K (R@K)
5. Mean Average Precisition (MAP)



[1] file:///Users/ofirziv/Downloads/Recommender-Systems-[Netflix].pdf
