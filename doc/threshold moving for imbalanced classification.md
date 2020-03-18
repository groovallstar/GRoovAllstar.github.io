# Threshold moving for imbalanced classification


<h2>Converting Probabilities to Class Labels</h2>
Many machine learning algorithms are capable of predicting a probability or a scoring of class membership.

많은 머신 러닝 알고리즘은 클래스 멤버쉽의 확률 또는 점수를 예측할 수 있습니다.

This is useful generally as it provides a measure of the certainty or uncertainty of a prediction. It also provides additional granularity over just predicting the class label that can be interpreted.

이것은 일반적으로 예측의 확실성 또는 불확실성의 척도를 제공하므로 유용합니다. 또한 해석할 수있는 클래스 레이블을 예측하는 것보다 세분화된 기능을 제공합니다.

Some classification tasks require a crisp class label prediction. This means that even though a probability or scoring of class membership is predicted, it must be converted into a crisp class label.

일부 분류 작업에는 선명한 클래스 레이블 예측이 필요합니다. 즉, 클래스 멤버십의 확률이나 점수가 예측 되더라도 선명한 클래스 레이블로 변환되어야합니다.

The decision for converting a predicted probability or scoring into a class label is governed by a parameter referred to as the “decision threshold,” “discrimination threshold,” or simply the “threshold.” The default value for the threshold is 0.5 for normalized predicted probabilities or scores in the range between 0 or 1.

예측 확률 또는 점수를 등급 레이블로 변환하기위한 결정은 "결정 임계 값", "차별 임계 값"또는 간단히 "임계 값"이라고하는 매개 변수에 의해 결정됩니다. 임계 값의 기본값은 정규화 된 예측 확률 또는 0에서 1 사이의 점수에 대해 0.5입니다.

For example, on a binary classification problem with class labels 0 and 1, normalized predicted probabilities and a threshold of 0.5, then values less than the threshold of 0.5 are assigned to class 0 and values greater than or equal to 0.5 are assigned to class 1.

예를 들어, 클래스 레이블이 0 및 1 인 이진 분류 문제에서 정규화 된 예측 확률과 임계 값이 0.5이면 임계 값 0.5보다 작은 값이 클래스 0에 할당되고 0.5보다 크거나 같은 값이 클래스 1에 할당됩니다.


    Prediction < 0.5 = Class 0
    Prediction >= 0.5 = Class 1

The problem is that the default threshold may not represent an optimal interpretation of the predicted probabilities.

문제는 기본 임계 값이 예측된 확률의 최적화 된 해석을 나타내지 않을 수 있다는 것입니다.

This might be the case for a number of reasons, such as:

다음과 같은 여러 가지 이유가 있을 수 있습니다.

    The predicted probabilities are not calibrated, e.g. those predicted by an SVM or decision tree.
    The metric used to train the model is different from the metric used to evaluate a final model.
    The class distribution is severely skewed.
    The cost of one type of misclassification is more important than another type of misclassification.

    예측된 확률은 교정되지 않습니다. 예 : SVM 또는 의사 결정 트리에 의해 예측 된 것.
    모델 학습에 사용되는 메트릭이 최종 모델을 평가하는 데 사용되는 메트릭과 다릅니다.
    클래스 분포가 심각하게 왜곡되었습니다.
    한 유형의 잘못 분류된 비용은 다른 유형의 잘못 분류된 것보다 더 중요합니다.

Worse still, some or all of these reasons may occur at the same time, such as the use of a neural network model with uncalibrated predicted probabilities on an imbalanced classification problem.

더 나쁜 것은, 불균형 분류 문제에 대해 교정되지 않은 예측 확률을 갖는 신경망 모델을 사용 하는 것에서 이런 이유 중 일부 또는 전부가 동시에 발생할 수 있습니다.

As such, there is often the need to change the default decision threshold when interpreting the predictions of a model.

따라서 종종 모델의 예측을 해석 할 때 기본 결정 임계 값을 변경해야 할 필요가 있습니다.

<h2>Threshold-Moving for Imbalanced Classification</h2>

There are many techniques that may be used to address an imbalanced classification problem, such as resampling the training dataset and developing customized version of machine learning algorithms.

훈련 데이터 세트를 리샘플링하고 기계 학습 알고리즘의 사용자 정의 버전을 개발하는 등 불균형 분류 문제를 해결하는 데 사용할 수 있는 많은 기술이 있습니다.

Nevertheless, perhaps the simplest approach to handle a severe class imbalance is to change the decision threshold. Although simple and very effective, this technique is often overlooked by practitioners and research academics alike as was noted by Foster Provost in his 2000 article titled “Machine Learning from Imbalanced Data Sets.”

그럼에도 불구하고 심각한 클래스 불균형을 처리하는 가장 간단한 방법은 의사 결정 임계 값을 변경하는 것입니다. 간단하고 매우 효과적이지만,이 기술은 Foster Provost가 2000 년에 쓴 “불균형 데이터 세트에서 기계 학습” 이라는 제목의 논문에서 언급 한 것처럼 실무자와 연구 학자들이 간과하는 경우가 많습니다.

The bottom line is that when studying problems with imbalanced data, using the classifiers produced by standard machine learning algorithms without adjusting the output threshold may well be a critical mistake.

결론은 불균형 데이터의 문제를 연구 할 때 출력 임계 값을 조정하지 않고 표준 기계 학습 알고리즘으로 생성된 분류기를 사용하는 것이 중대한 실수일 수 있다는 것입니다.

There are many reasons to choose an alternative to the default decision threshold.

기본 결정 임계 값에 대한 대안을 선택해야 하는 많은 이유가 있습니다.

For example, you may use ROC curves to analyze the predicted probabilities of a model and ROC AUC scores to compare and select a model, although you require crisp class labels from your model. How do you choose the threshold on the ROC Curve that results in the best balance between the true positive rate and the false positive rate?

예를 들어, 모델에서 선명한 클래스 레이블이 필요하지만 ROC 곡선을 사용하여 모델의 예측 확률과 ROC AUC 점수를 분석하여 모델을 비교하고 선택할 수 있습니다. ROC 곡선에서 true positive의 비율과 false positive의 비율 사이에서 최상의 균형을 이루는 임계 값을 어떻게 선택합니까?

Alternately, you may use precision-recall curves to analyze the predicted probabilities of a model, precision-recall AUC to compare and select models, and require crisp class labels as predictions. How do you choose the threshold on the Precision-Recall Curve that results in the best balance between precision and recall?

또는 precision-recall 곡선을 사용하여 모델의 예측 확률을 분석하고, precision-recall AUC를 사용하여 모델을 비교하고 선택하며, 선명한 클래스 레이블을 예측으로 요구할 수 있습니다. precision-recall 곡선에서 정밀도와 재현율 간에 최적의 균형을 이루는 임계 값을 어떻게 선택합니까?

You may use a probability-based metric to train, evaluate, and compare models like log loss (cross-entropy) but require crisp class labels to be predicted. How do you choose the optimal threshold from predicted probabilities more generally?

확률 기반 메트릭을 사용하여 로그 손실 (교차 엔트로피)과 같은 모델을 학습, 평가 및 비교할 수 있지만 선명한 클래스 레이블을 예측 해야 합니다. 예측 확률에서 더 일반적으로 최적의 임계 값을 어떻게 선택합니까?

Finally, you may have different costs associated with false positive and false negative misclassification, a so-called cost matrix, but wish to use and evaluate cost-insensitive models and later evaluate their predictions use a cost-sensitive measure. How do you choose a threshold that finds the best trade-off for predictions using the cost matrix?

마지막으로, 비용 매트릭스라고 불리는 false positive 및 false negative로 이뤄진 misclassification과 관련된 비용이 서로 다를 수 있지만 비용에 민감한 모델을 사용하고 평가 한 후 나중에 비용에 민감한 측정 값을 사용하여 예측을 평가하려고 합니다. 비용 매트릭스를 사용하여 예측에 가장 적합한 트레이드 오프를 찾는 임계 값을 어떻게 선택합니까?

    Popular way of training a cost-sensitive classifier without a known cost matrix is to put emphasis on modifying the classification outputs when predictions are being made on new data. This is usually done by setting a threshold on the positive class, below which the negative one is being predicted. The value of this threshold is optimized using a validation set and thus the cost matrix can be learned from training data.

    알려진 비용 매트릭스없이 비용에 민감한 분류기를 훈련시키는 일반적인 방법은 새로운 데이터에 대해 예측이 이루어질 때 분류 출력을 수정하는 데 중점을 두는 것입니다. 이것은 일반적으로 positive 클래스에서 임계 값을 설정하여 수행되며, 그 아래에서 negative 클래스가 예측됩니다. 이 임계 값은 유효성 검사 세트를 사용하여 최적화되므로 교육 데이터에서 비용 매트릭스를 학습 할 수 있습니다. 

— Page 67, Learning from Imbalanced Data Sets, 2018.

The answer to these questions is to search a range of threshold values in order to find the best threshold. In some cases, the optimal threshold can be calculated directly.

이러한 질문에 대한 답은 최상의 임계 값을 찾기 위해 임계 값 범위를 검색하는 것입니다. 경우에 따라 최적의 임계 값을 직접 계산할 수 있습니다.

Tuning or shifting the decision threshold in order to accommodate the broader requirements of the classification problem is generally referred to as “threshold-moving,” “threshold-tuning,” or simply “thresholding.”

분류 문제의 광범위한 요구 사항을 수용하기 위해 결정 임계 값을 조정하거나 변경하는 것을 일반적으로 "임계값 이동", "임계값 조정"또는 간단히 "임계값"이라고합니다. 

    It has been stated that trying other methods, such as sampling, without trying by simply setting the threshold may be misleading. The threshold-moving method uses the original training set to train [a model] and then moves the decision threshold such that the minority class examples are easier to be predicted correctly.

    단순히 임계값을 설정하지 않고 샘플링과 같은 다른 방법을 시도하면 오해의 소지가 있을 수 있습니다. 임계값 이동 방법은 기존 학습 세트를 사용하여 [model]을 학습 한 다음 소수 클래스 값을 정확하게 예측하기 쉽도록 결정 임계값을 이동합니다. 

— Pages 72, Imbalanced Learning: Foundations, Algorithms, and Applications, 2013.

The process involves first fitting the model on a training dataset and making predictions on a test dataset. The predictions are in the form of normalized probabilities or scores that are transformed into normalized probabilities. Different threshold values are then tried and the resulting crisp labels are evaluated using a chosen evaluation metric. The threshold that achieves the best evaluation metric is then adopted for the model when making predictions on new data in the future.

이 과정에는 먼저 훈련 데이터 세트에 모델을 맞추고 테스트 데이터 세트에 대한 예측을 수행합니다. 예측은 정규화된 확률 또는 정규화 된 확률로 변환 된 점수의 형태입니다. 그런 다음 다른 임계 값을 시도하고 결과로 선명한 레이블을 선택한 평가 메트릭을 사용하여 평가합니다. 그런 다음 향후 새로운 데이터를 예측할 때 최상의 평가 메트릭을 달성하는 임계 값이 모델에 적용됩니다.

We can summarize this procedure below.

    1. Fit Model on the Training Dataset.
    2. Predict Probabilities on the Test Dataset.
    3. For each threshold in Thresholds:
        3a. Convert probabilities to Class Labels using the threshold.
        3b. Evaluate Class Labels.
        3c. If Score is Better than Best Score.
            3ci. Adopt Threshold.
        4. Use Adopted Threshold When Making Class Predictions on New Data.

Although simple, there are a few different approaches to implementing threshold-moving depending on your circumstance. We will take a look at some of the most common examples in the following sections.


<h2>Optimal Threshold for ROC Curve</h2>

A ROC curve is a diagnostic plot that evaluates a set of probability predictions made by a model on a test dataset.

ROC 곡선은 테스트 데이터 세트에서 모델에 의해 만들어진 확률 예측 세트를 평가하는 진단 plot입니다. 

A set of different thresholds are used to interpret the true positive rate and the false positive rate of the predictions on the positive (minority) class, and the scores are plotted in a line of increasing thresholds to create a curve.

positive (소수) 클래스에 대한 예측의 true positive rate와 false positive rate를 해석하기 위해 서로 다른 임계값 세트가 사용되며, 점수는 임계 값이 증가하는 라인에 plot 되어 곡선을 만듭니다. 

The false-positive rate is plotted on the x-axis and the true positive rate is plotted on the y-axis and the plot is referred to as the Receiver Operating Characteristic curve, or ROC curve. A diagonal line on the plot from the bottom-left to top-right indicates the “curve” for a no-skill classifier (predicts the majority class in all cases), and a point in the top left of the plot indicates a model with perfect skill.

false-positive rate은 x 축에 표시되고 true positive rate은 y 축에 표시되며 plot은 Receiver Operating Characteristic curve 또는 ROC 곡선이라고합니다. 왼쪽 하단에서 오른쪽 상단에있는 plot의 대각선은 no-skill 분류기의 "곡선"을 나타내고 (모든 경우에 대부분의 클래스를 예측 함) plot의 왼쪽 상단에 있는 점은 perfect-skill의 모델을 나타냅니다. 

The curve is useful to understand the trade-off in the true-positive rate and false-positive rate for different thresholds. The area under the ROC Curve, so-called ROC AUC, provides a single number to summarize the performance of a model in terms of its ROC Curve with a value between 0.5 (no-skill) and 1.0 (perfect skill).

이 곡선은 다른 임계 값에 대한 true-positive rate과 false-positive rate의 절충을 이해하는 데 유용합니다. ROC AUC라고하는 ROC 곡선 아래 영역은 0.5 (no-skill)에서 1.0 (perfect skill) 사이의 값으로 ROC 곡선 측면에서 모델의 성능을 요약하는 단일 숫자를 제공합니다.

The ROC Curve is a useful diagnostic tool for understanding the trade-off for different thresholds and the ROC AUC provides a useful number for comparing models based on their general capabilities.

ROC 곡선은 다양한 임계 값에 대한 절충을 이해하는 데 유용한 진단 도구이며 ROC AUC는 일반 기능을 기반으로 모델을 비교하는 데 유용한 수를 제공합니다. 

If crisp class labels are required from a model under such an analysis, then an optimal threshold is required. This would be a threshold on the curve that is closest to the top-left of the plot.

이러한 모델의 선명한 클래스 레이블이 요구 되는 분석에서는 최적의 임계 값이 필요합니다. 이것은 plot의 왼쪽 상단에 가장 가까운 곡선의 임계 값이 되려고 할 것 입니다. 

Thankfully, there are principled ways of locating this point.

First, let’s fit a model and calculate a ROC Curve.

We can use the make_classification() function to create a synthetic binary classification problem with 10,000 examples (rows), 99 percent of which belong to the majority class and 1 percent belong to the minority class.

<pre>
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)

...
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

...
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)

...
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

...
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
</pre>
We can then use the roc_auc_score() function to calculate the true-positive rate and false-positive rate for the predictions using a set of thresholds that can then be used to create a ROC Curve plot.

<pre>...
# calculate scores
lr_auc = roc_auc_score(testy, lr_probs)
</pre>

<pre>
# roc curve for logistic regression model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
fpr, tpr, thresholds = roc_curve(testy, yhat)
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()
</pre>

Running the example fits a logistic regression model on the training dataset then evaluates it using a range of thresholds on the test set, creating the ROC Curve

예제를 실행하면 훈련 데이터 세트에 로지스틱 회귀 모델이 적합하고 테스트 세트에서 임계 값 범위를 사용하여 ROC 곡선을 작성하여 모델을 평가합니다. 

We can see that there are a number of points or thresholds close to the top-left of the plot.

plot의 왼쪽 상단에 많은 포인트 또는 임계 값이 있음을 알 수 있습니다. 

Which is the threshold that is optimal?

최적의 임계 값은 어느 것입니까?

[<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/11/ROC-Curve-Line-Plot-for-Logistic-Regression-Model-for-Imbalanced-Classification-1024x768.png">](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)

There are many ways we could locate the threshold with the optimal balance between false positive and true positive rates.

false positive rate과 true positive rate 사이의 최적 균형으로 임계 값을 찾을 수 있는 방법은 여러 가지가 있습니다. 

Firstly, the true positive rate is called the Sensitivity. The inverse of the false-positive rate is called the Specificity.

첫째, true positive rate을 민감도라고합니다. false-positive rate의 역수를 특이성이라고합니다.

    Sensitivity = TruePositive / (TruePositive + FalseNegative)
    Specificity = TrueNegative / (FalsePositive + TrueNegative)

Where:

    Sensitivity = True Positive Rate
    Specificity = 1 – False Positive Rate

The Geometric Mean or G-Mean is a metric for imbalanced classification that, if optimized, will seek a balance between the sensitivity and the specificity.

기하 평균 또는 G-Mean은 최적화된 경우의 민감도와 특이성의 균형을 찾는 불균형 분류에 대한 지표입니다.

    G-Mean = sqrt(Sensitivity * Specificity)

One approach would be to test the model with each threshold returned from the call roc_auc_score() and select the threshold with the largest G-Mean value.

한 가지 방법은 roc_auc_score() 호출에서 반환된 각 임계 값으로 모델을 테스트하고 G-Mean 값이 가장 큰 임계 값을 선택하는 것입니다. 

Given that we have already calculated the Sensitivity (TPR) and the complement to the Specificity when we calculated the ROC Curve, we can calculate the G-Mean for each threshold directly.

ROC 곡선을 계산할 때 이미 민감도 (TPR)와 특이성에 대한 보완을 계산 했으므로 각 임계값에 대한 G-Mean을 직접 계산할 수 있습니다.

<pre>
# roc curve for logistic regression model with optimal threshold
from numpy import sqrt
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
fpr, tpr, thresholds = roc_curve(testy, yhat)
# calculate the g-mean for each threshold
gmeans = sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()
</pre>

[<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/11/ROC-Curve-Line-Plot-for-Logistic-Regression-Model-for-Imbalanced-Classification-with-the-Optimal-Threshold-1024x768.png">](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)

Running the example first locates the optimal threshold and reports this threshold and the G-Mean score.

예제를 먼저 실행하면 최적의 임계 값을 찾고이 임계 값과 G-Mean 점수를 보고합니다. 

In this case, we can see that the optimal threshold is about 0.016153.
Best Threshold=0.016153, G-Mean=0.933

The threshold is then used to locate the true and false positive rates, then this point is drawn on the ROC Curve.

그런 다음 임계 값을 사용하여 true positive rate 및 false positive rate을 찾은 다음이 점을 ROC 곡선에 그립니다. 

We can see that the point for the optimal threshold is a large black dot and it appears to be closest to the top-left of the plot.

최적의 임계 값에 대한 점이 큰 검은 점이며 플롯의 왼쪽 상단에 가장 가까운 것으로 보입니다.

It turns out there is a much faster way to get the same result, called the Youden’s J statistic.

Youden의 J 통계라고 불리는 동일한 결과를 얻는 훨씬 빠른 방법이 있습니다.

The statistic is calculated as:

    J = Sensitivity + Specificity – 1

Given that we have Sensitivity (TPR) and the complement of the specificity (FPR), we can calculate it as:

    J = Sensitivity + (1 – FalsePositiveRate) – 1

Which we can restate as:

    J = TruePositiveRate – FalsePositiveRate

We can then choose the threshold with the largest J statistic value. For example:

<pre>
# roc curve for logistic regression model with optimal threshold
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
fpr, tpr, thresholds = roc_curve(testy, yhat)
# get the best threshold
J = tpr - fpr
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f' % (best_thresh))
</pre>

We can see that this simpler approach calculates the optimal statistic directly.
Best Threshold=0.016153

<h2> Optimal Threshold for Precision-Recall Curve </h2>

Unlike the ROC Curve, a precision-recall curve focuses on the performance of a classifier on the positive (minority class) only.

ROC 곡선과 달리 precision-recall 곡선은 positive(소수 등급)에 대해서만 분류기의 성능에 중점을 둡니다.

Precision is the ratio of the number of true positives divided by the sum of the true positives and false positives. It describes how good a model is at predicting the positive class. Recall is calculated as the ratio of the number of true positives divided by the sum of the true positives and the false negatives. Recall is the same as sensitivity.

정밀도는 true positives 수를 true positive및 false positive의 합으로 나눈 비율입니다. 모델이 양성 클래스를 얼마나 잘 예측하는지 설명합니다. 재현율은 true positive 수를 true positive 및 false negative의 합으로 나눈 비율로 계산됩니다. 재현율은 감도와 동일합니다. 

A precision-recall curve is calculated by creating crisp class labels for probability predictions across a set of thresholds and calculating the precision and recall for each threshold. A line plot is created for the thresholds in ascending order with recall on the x-axis and precision on the y-axis.

precision-recall 곡선은 임계 값 세트에 걸쳐 확률 예측을 위한 선명한 클래스 레이블을 생성하고 각 임계 값에 대한 정밀도 및 리콜을 계산하여 계산됩니다. x 축의 리콜과 y 축의 정밀도로 오름차순으로 임계 값에 대한 선 그림이 생성됩니다. 

A no-skill model is represented by a horizontal line with a precision that is the ratio of positive examples in the dataset (e.g. TP / (TP + TN)), or 0.01 on our synthetic dataset. perfect skill classifier has full precision and recall with a dot in the top-right corner.

no-skill 모델은 데이터 세트의 positive 예 (예 : TP / (TP + TN)) 또는 합성 데이터 세트의 0.01 인 정밀도의 가로 선으로 표시됩니다. perfect skill 분류기는 전체 정밀도를 가지며 오른쪽 상단 모서리에 점이 있습니다. 

We can use the same model and dataset from the previous section and evaluate the probability predictions for a logistic regression model using a precision-recall curve. The precision_recall_curve() function can be used to calculate the curve, returning the precision and recall scores for each threshold as well as the thresholds used.

이전 섹션의 동일한 모델과 데이터 집합을 사용하고 정밀 회귀 곡선을 사용하여 로지스틱 회귀 모델의 확률 예측을 평가할 수 있습니다. precision_recall_curve() 함수는 곡선을 계산하는 데 사용되어 각 임계 값과 사용 된 임계 값에 대한 정밀도 및 재현율 점수를 반환합니다.

<pre>
# pr curve for logistic regression model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate pr-curve
precision, recall, thresholds = precision_recall_curve(testy, yhat)
# plot the roc curve for the model
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
# show the plot
pyplot.show()
</pre>

Running the example calculates the precision and recall for each threshold and creates a precision-recall plot showing that the model has some skill across a range of thresholds on this dataset.

예제를 실행하면 각 임계 값에 대한 정밀도 및 재현율이 계산되고 모델이 이 데이터 세트의 임계 값 범위에서 약간의 기술을 가지고 있음을 보여주는 precision-recall 그림이 생성됩니다.

If we required crisp class labels from this model, which threshold would achieve the best result?

이 모델에서 선명한 클래스 레이블이 필요한 경우 어떤 임계 값이 가장 좋은 결과를 얻습니까?

[<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2020/02/Precision-Recall-Curve-Line-Plot-for-Logistic-Regression-Model-for-Imbalanced-Classification2-1024x768.png">](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)

If we are interested in a threshold that results in the best balance of precision and recall, then this is the same as optimizing the F-measure that summarizes the harmonic mean of both measures.

최고의 정밀도와 리콜의 균형을 이끌어내는 임계 값에 관심이 있다면, 이는 두 측정의 조화 평균을 요약 한 F-measure을 최적화하는 것과 같습니다. 

    F-Measure = (2 * Precision * Recall) / (Precision + Recall)

As in the previous section, the naive approach to finding the optimal threshold would be to calculate the F-measure for each threshold. We can achieve the same effect by converting the precision and recall measures to F-measure directly; for example:

이전 섹션에서와 같이 최적 임계 값을 찾는 순진한 접근 방식은 각 임계 값에 대한 F-measure 값을 계산하는 것입니다. 정밀도 및 호출 측정 값을 F-measure 값으로 직접 변환하여 동일한 효과를 얻을 수 있습니다. 예를 들면 다음과 같습니다.

<pre>
# optimal threshold for precision-recall curve with logistic regression model
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
precision, recall, thresholds = precision_recall_curve(testy, yhat)
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
# plot the roc curve for the model
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', label='Logistic')
pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
# show the plot
pyplot.show()
</pre>

Running the example first calculates the F-measure for each threshold, then locates the score and threshold with the largest value.

예제를 먼저 실행하면 각 임계 값에 대한 F-measure 값을 계산 한 다음 가장 큰 값으로 점수와 임계 값을 찾습니다. 

In this case, we can see that the best F-measure was 0.756 achieved with a threshold of about 0.25.

    Best Threshold=0.256036, F-Score=0.756

The precision-recall curve is plotted, and this time the threshold with the optimal F-measure is plotted with a larger black dot.

precision-recall curve가 그려지고, 이번에는 최적의 F-measure가 있는 임계 값이 더 큰 검은 점으로 그려집니다. 

This threshold could then be used when making probability predictions in the future that must be converted from probabilities to crisp class labels.

이 임계 값은 이후 확률에서 선명한 클래스 레이블로 변환 되어야 하는 확률 예측을 수행 할 때 사용될 수 있습니다.

[<img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2020/02/Precision-Recall-Curve-Line-Plot-for-Logistic-Regression-Model-With-Optimal-Threshold2-1024x768.png">](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)

<h2> Optimal Threshold Tuning </h2>

Sometimes, we simply have a model and we wish to know the best threshold directly.

때때로, 우리는 단순히 모델을 가지고 있으며 가장 좋은 임계 값을 직접 알고 싶어합니다. 

In this case, we can define a set of thresholds and then evaluate predicted probabilities under each in order to find and select the optimal threshold.

이 경우 최적의 임계 값을 찾아서 선택하기 위해 임계 값 세트를 정의한 후 각각의 예상 확률을 평가할 수 있습니다. 

We can demonstrate this with a worked example.

우리는 이것을 실례로 보여줄 수 있습니다. 

First, we can fit a logistic regression model on our synthetic classification problem, then predict class labels and evaluate them using the F-Measure, which is the harmonic mean of precision and recall.

먼저, 합성 분류 문제에 로지스틱 회귀 모델을 적용한 다음 클래스 레이블을 예측하고 F-Measure (F-Measure)를 사용하여 이를 평가하고 평가할 수 있습니다. 

This will use the default threshold of 0.5 when interpreting the probabilities predicted by the logistic regression model.

로지스틱 회귀 모형으로 예측 된 확률을 해석 할 때 기본 임계 값 인 0.5를 사용합니다. 

The complete example is listed below.

<pre>
# search thresholds for imbalanced classification
from numpy import arange
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# define thresholds
thresholds = arange(0, 1, 0.001)
# evaluate each threshold
scores = [f1_score(testy, to_labels(probs, t)) for t in thresholds]
# get best threshold
ix = argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
</pre>

Running the example reports the optimal threshold as 0.251 (compared to the default of 0.5) that achieves an F-Measure of about 0.75 (compared to 0.70).

예제를 실행하면 약 0.75 (0.70과 비교)의 F-Measure을 달성하는 최적 임계 값을 0.251 (기본값 0.5와 비교)로 나타내줍니다.

You can use this example as a template when tuning the threshold on your own problem, allowing you to substitute your own model, metric, and even resolution of thresholds that you want to evaluate.

자신의 문제 임계 값을 조정할 때 이 예제를 템플릿으로 사용하여 평가하는 임계 값의 고유 모델 메트릭, 심지어 resolution of thresholds를 바꿀 수 있습니다.

Threshold=0.251, F-Score=0.75556

# Reference.
https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
