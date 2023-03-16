# [PyTorch] Recommender system 
## Introduction
This project is evaluate the quality of the AutoEncoder model on [Movielens-1M dataset](https://github.com/windhashira06/RS-AutoEncoder/tree/main/movielens-1m) and suggest to the user which movies they might like.
## Datasets
The Movielens 1M dataset is a popular dataset in the fields of Machine Learning and Data Mining. This dataset consists of 1 million movie ratings rated by 6000 users with over 4000 different movies. This dataset is devided into 2 datasets, that is the train set (750121 ratings) and the test set (250088 ratings).
## Model
 The AutoEncoder model used in this task consists of 1 input layer, 3 hidden layers and 1 output layer. The number of hidden units in each hidden layer are 200, 100 and 200 respectively.
<p align="center">
  <img src="https://user-images.githubusercontent.com/121651344/225369029-74d69875-d8da-48b8-a046-bfd8c7f05b90.png" alt="autoencoder">
</p>

## Training and Evaluation
__1. Training__: The model's input is a utility matrix containing values from 0 to 5, where 0 is the value representing movies that have not been rated by the user and 1 to 5 are user-rated values. The model will learn to represent user-movies as latents vector in the new feature space from which the model will predictions for empty positions (value is 0) in the utility matrix. 
```
for user_nb in range(self.num_users):
    inputs = Variable(train_ds[user_nb]).unsqueeze(0).to(device)
    target = inputs.clone().to(device)
    mask = (target != 0).type(torch.float)*1.
    mask = mask.to(device)
                
    if torch.sum(target.data > 0) > 0:
    outputs = self.model(inputs, self.act_mode)
                    
    target.requires_grad = False
    mask.requires_grad = False

    loss = self.loss(outputs, target, mask)
    loss.backward()
    train_loss += loss.item()
    s += 1
    self.optimizer.step()
```
   The model takes each user-movies vector as input, computes output, compute loss and optimizer                                              
   - Note: We need to multiply the output matrix by a mask matrix to ensure that any positions of the target matrix are 0, those positions of the output matrix are also 0.            

__2. Evaluation__: To evaluate model quality, I use RMSE loss and Top-k Accuracy. Below are the evaluation results:
- RMSE loss:

  | RMSE training loss | RMSE validation loss |
  | :----------------: | :------------------: |
  |       0.88847      |        0.94895       |
- Top-k Accuracy:

  | K = 5 | K = 10 |
  | :----:|:------:|
  | 69.8725| 77.4966|
## Give Recommendations
The model will restore the missing values in the utility matrix, then select k (optional) positions containing the largest values (the larger value, the more user will like that movie) corresponding to k-movies's id that the user will probably like very much.         
Below are some recommended results:              
```
==> ID USER:  8
==> RECOMMEND: 
   MOVIE ID                MOVIE NAME                MOVIE GENRE
0       317  Santa Clause, The (1994)  Children's|Comedy|Fantasy
1      2196          Knock Off (1998)                     Action
2      2761    Iron Giant, The (1999)       Animation|Children's
3      2904               Rain (1932)                      Drama
4      3232      Seven Chances (1925)                     Comedy
```
```
==> ID USER:  9
==> RECOMMEND: 
   MOVIE ID                         MOVIE NAME     MOVIE GENRE
0       295  Pyromaniac's Love Story, A (1995)  Comedy|Romance
1      1078                     Bananas (1971)      Comedy|War
2      1192            Paris Is Burning (1990)     Documentary
3      2904                        Rain (1932)           Drama
4      3232               Seven Chances (1925)          Comedy
```
```
==> ID USER:  13
==> RECOMMEND: 
   MOVIE ID                MOVIE NAME                MOVIE GENRE
0       259      Kiss of Death (1995)       Crime|Drama|Thriller
1       317  Santa Clause, The (1994)  Children's|Comedy|Fantasy
2      2761    Iron Giant, The (1999)       Animation|Children's
3      3232      Seven Chances (1925)                     Comedy
```
```
==> ID USER:  15
==> RECOMMEND: 
   MOVIE ID                                MOVIE NAME           MOVIE GENRE
0       526  Savage Nights (Nuits fauves, Les) (1992)                 Drama
1      2323                        Cruise, The (1998)           Documentary
2      2761                    Iron Giant, The (1999)  Animation|Children's
3      3469                   Inherit the Wind (1960)                 Drama
4      3915                          Girlfight (2000)                 Drama
```
```
==> ID USER:  18
==> RECOMMEND: 
   MOVIE ID            MOVIE NAME MOVIE GENRE
0      2196      Knock Off (1998)      Action
1      2904           Rain (1932)       Drama
2      3091      Kagemusha (1980)   Drama|War
3      3232  Seven Chances (1925)      Comedy
```
```
==> ID USER:  20
==> RECOMMEND: 
   MOVIE ID            MOVIE NAME               MOVIE GENRE
0      1248  Touch of Evil (1958)  Crime|Film-Noir|Thriller
1      2904           Rain (1932)                     Drama
2      3091      Kagemusha (1980)                 Drama|War
3      3232  Seven Chances (1925)                    Comedy
4      3446    Funny Bones (1995)                    Comedy
```

Thank you a lot for the finding! 😊
