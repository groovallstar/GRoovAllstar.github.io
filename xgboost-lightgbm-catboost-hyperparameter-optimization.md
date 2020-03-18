# An Example of Hyperparameter Optimization on XGBoost, LightGBM and CatBoost using Hyperopt


<h2>Gradient Boosting Decision Tree (GBDT)</h2>
Gradient Boosting is an additive training technique on Decision Trees. The official page of XGBoost gives a very clear explanation of the concepts. Basically, instead of running a static single Decision Tree or Random Forest, new trees are being added iteratively until no further improvement can be achieved. The ensembling technique in addition to regularization are critical in preventing overfitting. Although the model could be very powerful, a lot of hyperparamters are there to be fine-tuned.

그라디언트 부스팅은 의사 결정 트리에 대한 추가 교육 기술입니다. XGBoost의 공식 페이지는 개념에 대한 명확한 설명을 제공합니다. 기본적으로 정적 single Decision Tree 또는 Random Forest를 실행하는 대신 더 이상 개선 할 수 없을 때까지 새 트리가 반복적으로 추가됩니다. 정규화 외에도 ensemble 기술은 과적 합을 방지하는 데 중요합니다. 모델은 매우 강력 할 수 있지만 많은 하이퍼 파라미터가 미세 조정 되어야 합니다.

<h2> XGBoost, LightGBM, and CatBoost </h2>

These are the well-known packages for gradient boosting. Compared with the traditional GBDT approach which finds the best split by going through all features, these packages implement histogram-based method that groups features into bins and perform splitting at the bin level rather than feature level. On the other hand they tend to ignore sparse inputs as well. These significantly improve their computational speed (see here for more detail). Few more keys to note:

그라디언트 부스팅을 위해 잘 알려진 패키지입니다. 모든 기능을 통해 최상의 분할을 찾는 기존 GBDT 방식과 비교하여 이 패키지는 기능을 빈으로 그룹화하고 기능 수준이 아닌 빈 수준에서 분할을 수행하는 히스토그램 기반 방법을 구현합니다. 반면에 그들은 희소 입력도 무시하는 경향이 있습니다. 이들은 계산 속도를 크게 향상시킵니다 (자세한 내용은 여기 참조). 참고해야 할 키가 더 적습니다.

XGBoost: The famous Kaggle winning package. Tree growing is based on level-wise tree pruning (tree grows across all node at a level) using the information gain from spliting, for which the samples need to be pre-sorted for it to calculate the best score across all possible splits in each step and thus is comparatively time-consuming.

XGBoost : 유명한 Kaggle 우승 패키지. 트리 성장은 분할에서 얻은 정보 획득을 사용하여 레벨 별 트리 가지 치기 (트리가 모든 노드에서 레벨에 따라 성장)를 기반으로 합니다.이 경우 각 샘플에서 가능한 모든 분할에 대해 최상의 점수를 계산하기 위해 샘플을 미리 정렬해야합니다. 단계적으로 시간이 많이 소요됩니다.

LightGBM: Both level-wise and leaf-wise (tree grows from particular leaf) training are available. It allows user to select a method called Gradient-based One-Side Sampling (GOSS) that splits the samples based on the largest gradients and some random samples with smaller gradients. The assumption behind is that data points with smaller gradients are more well-trained. Another key algorithm is the Exclusive Feature Bundling (EFB), which looks into the sparsity of features and combines multiple features into one without losing any information given that they are never non-zero together. These makes LightGBM a speedier option compared to XGBoost.

LightGBM : level-wise 및 leaf-wise (나무는 특정 잎에서 자랍니다) train을 사용할 수 있습니다. 사용자는 가장 큰 그래디언트와 더 작은 그래디언트를 가진 임의의 샘플을 기반으로 샘플을 분할하는 Gradient-based One-Side Sampling (GOSS)이라는 방법을 선택할 수 있습니다. 배후가 작은 가정은 더 작은 그라디언트를 가진 데이터 포인트가 더 잘 훈련된다는 것입니다. 또 다른 주요 알고리즘은 EFB (Exclusive Feature Bundling)입니다.이 기능은 특징의 희소성을 조사하고 0이 아닌 정보를 잃지 않고 여러 기능을 하나로 결합합니다. 따라서 XGBoost에 비해 LightGBM이 더 빨라졌습니다.

CatBoost: Specifically designed for categorical data training, but also applicable to regression tasks. The speed on GPU is claimed to be the fastest among these libraries. It has various methods in transforming catergorical features to numerical. The keys to its speed are linked to two Os: Oblivious Tree and Ordered Boosting. Oblivious Tree refers to the level-wised tree building with symmetric binary splitting (i.e each leaf on each level are split by a single feature), while Ordered Boosting applies permutation and sequential target encoding to convert categorical features. See here and here for more details.

CatBoost : 범주형 데이터 교육용으로 특별히 설계되었지만 회귀 작업에도 적용 할 수 있습니다. GPU 속도는 이러한 라이브러리 중에서 가장 빠르다고 주장합니다. 그것은 catergorical 기능을 수치로 변환하는 다양한 방법을 가지고 있습니다. 속도의 열쇠는 Oblivious Tree와 Ordered Boosting의 두 가지 Os에 연결되어 있습니다. Oblivious Tree는 대칭 이진 분할 (즉, 각 수준의 각 리프가 단일 기능으로 분할 됨)을 사용하는 수준별 트리 건물을 의미하는 반면, Ordered Boosting은 순열 및 순차 대상 인코딩을 적용하여 범주 형 기능을 변환합니다. 자세한 내용은 여기와 여기를 참조하십시오.

<h2>Bayesian Optimization</h2>

Compared with GridSearch which is a brute-force approach, or RandomSearch which is purely random, the classical Bayesian Optimization combines randomness and posterior probability distribution in searching the optimal parameters by approximating the target function through Gaussian Process (i.e. random samples are drawn iteratively (Sequential Model-Based Optimization (SMBO)) and the function outputs between the samples are approximated by a confidence region). New samples will be drawn from the parameter space at the high mean and variance over the confidence region for exploration and exploitation. Check this out for more explanation.

# Reference.
https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e
