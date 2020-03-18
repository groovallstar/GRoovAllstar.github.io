
# feature selection
http://hero4earth.com/blog/learning/2018/01/29/Feature_Engineering_Basic/
https://www.youtube.com/watch?v=G2QrVM_V5uU&list=PL2uF2b2friV594ijimTrgwk9dzl8_3dTe&index=6
https://www.youtube.com/watch?v=iWnt1_js1zM

이번엔 Feature Importance 값 자체를 보겠습니다. 전체 값들의 간단한 통계 수치는 다음과 같습니다.

    전체 수 : 4,903
    평균 : 1.2033
    분산 : 94.4078
    최소값 : 0
    최대값 : 347.4
    중앙값 : 0

중앙값이 0이라는 점이 눈에 띕니다. Feature 중 50% 이상의 Importance가 0이라는 얘기입니다. 
확인을 해보았더니 857개를 제외하고 나머지 82%에 해당하는 4,046개 Feature의 Importance가 모두 0이었습니다. 
모델이 이렇게 많은 Feature를 중요하지 않다고 판단했다면, 굳이 모든 Feature들을 사용할 필요가 없어보입니다. 
Feature 일부를 제거하여 학습하면 계산량과 메모리 사용량을 줄이고 더 빠르게 학습 결과를 도출할 수 있을 것으로 예상됩니다.

Feature Selection

Feature Selection은 성능에 더 많은 영향을 주는 Feature를 우선적으로 선택하여 학습하는 방법입니다. 
상위 몇 개 혹은 몇 %를 선택하는 방법, 반복적으로 하위 Feature를 제거해 나가는 방법 등 다양한 방법이 존재합니다.
우선 LightGBM모델에서 Feature Importance가 0인 값만 제거하여 실험해보았습니다. 
성능에는 큰 영향을 주지 않았지만, 전체 중 82%의 무의미한 Feature의 값이 제거되었기 때문에 LightGBM이 학습해야 하는 Feature의 수가 줄어들면서 학습 속도가 3배 이상 향상했습니다. 
이번에는 학습된 LightGBM의 Feature Importance를 이용하여 선택된 Feature를 Neural Network 모델에도 적용하여 실험해보았습니다. 
Neural Network는 Hidden Layer 때문에 Feature Importance를 쉽게 추출할 수 없으므로 다른 모델의 Feature 정보를 이용하면 성능을 더 올릴 수 있지 않을까하는 생각이었습니다.

마지막으로는 classifier의 성능 측정을 할 때에는 Brier score, Log loss같은 proper scoring rule을 사용하는것이 사실은 맞습니다만 
대부분 모르는 경우가 있어서 cut-off를 조정하여 찾은 뒤 accuracy, precision, recall 등과 병기하거나 아니면 cut-off에 민감하지 않은 auroc, auprc, f1 등을 사용하셔서 분류기를 평가하는 것이 낫겠습니다.

# Binary Classifier Accuracy
http://www-stat.wharton.upenn.edu/~buja/PAPERS/paper-proper-scoring.pdf
https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models
https://www.fharrell.com/post/class-damage/
https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
https://stackoverflow.com/questions/53846943/sklearn-logistic-regression-adjust-cutoff-point

# Classification on imbalanced data
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

# Permutation_importance
https://explained.ai/rf-importance/
https://scikit-learn.org/stable/modules/permutation_importance.html

# ROC-AUC
https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221015817276&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F

# Bias-Variance
http://scott.fortmann-roe.com/docs/BiasVariance.html

# Overfitting
https://elitedatascience.com/overfitting-in-machine-learning

# Brier Score
SKLearn
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
https://scikit-learn.org/stable/modules/calibration.html#calibration
https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
What is a Brier Score?
https://www.statisticshowto.datasciencecentral.com/brier-score/
결정 트리의 지니 불순도란 무엇일까?
https://smalldataguru.com/%EA%B2%B0%EC%A0%95-%ED%8A%B8%EB%A6%ACdecision-tree%EC%9D%98-%EC%A7%80%EB%8B%88-%EB%B6%88%EC%88%9C%EB%8F%84gini-impurity%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%BC%EA%B9%8C/