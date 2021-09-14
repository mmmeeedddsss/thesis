

## Metrics

### Balanced testset split evals

##### surprise.SVD
```
MAE:  1.0725
MSE: 2.22934761
RMSE: 1.4931
```

##### My algorithm(KeyBert -> Word2vec -> DecisionTree):
```
MAE for iter 0: 1.1252525252525252
MSE for iter 0: 2.9927272727272727
RMSE for iter 0: 1.7299500781026234
```

##### My algorithm(KeyBert -> Word2vec -> DecisionTree) + IDF Scale:
```
MAE for iter 0: 1.166595105195363
MSE for iter 0: 3.1562902533276085
RMSE for iter 0: 1.7765951292648554
```


##### My algorithm(Yake -> Word2vec -> DecisionTree):
```
MAE for iter 0: 1.1556184316895715
MSE for iter 0: 3.09498787388844
RMSE for iter 0: 1.7592577622078125
```

##### My algorithm(TfIdf -> Word2vec -> DecisionTree):
```
MAE for iter 0: 1.2724333063864188
MSE for iter 0: 3.0202101859337107
RMSE for iter 0: 1.7378751928529594
```

##### My algorithm(Yake -> Word2vec -> OrdinalDecisionTree):
```
MAE for iter 0: 1.3172999191592563
MSE for iter 0: 3.2809215844785773
RMSE for iter 0: 1.8113314397090825
```

##### My algorithm(KeyBert -> Word2vec -> OrdinalDecisionTree):
```
MAE for iter 0: 1.3427647534357317
MSE for iter 0: 3.2344381568310427
RMSE for iter 0: 1.7984543799693788
```

##### Prediction as mean rating on train set:
```
Baseline MAE for iter 0: 1.9995553579368608
Baseline MSE for iter 0: 5.998666073810583
Baseline RMSE for iter 0: 2.4492174411045218
```
