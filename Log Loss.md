# What is Log Loss?


<h2>Introduction</h2>

Log Loss is the most important classification metric based on probabilities.

로그 손실은 확률을 기반으로 가장 중요한 분류 지표입니다.

It's hard to interpret raw log-loss values, but log-loss is still a good metric for comparing models. For any given problem, a lower log-loss value means better predictions.

raw 로그 손실 값을 해석하기는 어렵지만 로그 손실은 여전히 ​​모델을 비교하기에 좋은 지표입니다. 주어진 문제에 대해 로그 손실 값이 낮을수록 더 나은 예측을 의미합니다.

Log Loss is a slight twist on something called the Likelihood Function. In fact, Log Loss is -1 * the log of the likelihood function. So, we will start by understanding the likelihood function.

로그 손실은 우도 함수 (Likelihood Function)라고 불리는 것에 약간의 왜곡이 있습니다. 실제로 Log Loss는 우도 함수의 log * -1 입니다. 따라서 우도 함수를 이해하는 것으로 시작하겠습니다.

The likelihood function answers the question "How likely did the model think the actually observed set of outcomes was." If that sounds confusing, an example should help.

우도 함수는 "모델이 실제로 예측된 결과 셋을 어느 정도 생각 했는가?" 라는 질문에 대답합니다. 혼란스럽게 들리면 예제가 도움이 될 것입니다.

<h2>Example</h2>

A model predicts probabilities of [0.8, 0.4, 0.1] for three houses. The first two houses were sold, and the last one was not sold. So the actual outcomes could be represented numeically as [1, 1, 0].

모델은 세 집에 대해 [0.8, 0.4, 0.1]의 확률을 예측합니다. 처음 두 집은 팔렸고 마지막 집은 팔리지 않았습니다. 따라서 실제 결과는 수치 적으로 [1, 1, 0]으로 표시 될 수 있습니다.

Let's step through these predictions one at a time to iteratively calculate the likelihood function.

우도 함수를 반복적으로 계산하기 위해 이러한 예측을 한 번에 하나씩 살펴 보겠습니다.

The first house sold, and the model said that was 80% likely. So, the likelihood function after looking at one prediction is 0.8.

첫 번째 집이 팔렸고 모델은 80% 가능성이 있다고 말했다. 따라서 우도 함수의 예측값은 0.8 입니다.

The second house sold, and the model said that was 40% likely. There is a rule of probability that the probability of multiple independent events is the product of their individual probabilities. So, we get the combined likelihood from the first two predictions by multiplying their associated probabilities. That is 0.8 * 0.4, which happens to be 0.32.

두 번째 집은 팔렸고 모델은 40% 가능성이 있다고 말했다. 여러 독립 사건의 확률이 개별 확률의 곱일 확률에 대한 규칙이 있습니다. 따라서 우리는 관련된 두 확률을 곱하여 처음 두 예측에서 결합 된 가능성을 얻습니다. 0.8 * 0.4 이며 0.32입니다.

Now we get to our third prediction. That home did not sell. The model said it was 10% likely to sell. That means it was 90% likely to not sell. So, the observed outcome of not selling was 90% likely according to the model. So, we multiply the previous result of 0.32 by 0.9.
We could step through all of our predictions. Each time we'd find the probability associated with the outcome that actually occurred, and we'd multiply that by the previous result. That's the likelihood.

이제 세번째 예측에 도달했습니다. 그 집은 팔리지 않았다. 이 모델은 팔릴 가능성이 10%라고 밝혔다. 즉, 팔리지 않을 확률이 90% 라는 것을 의미합니다. 따라서 모델에 따르면, 팔리지 않는다는 예측 결과는 90%가 될 수 있다. 그래서 이전 결과 0.32에 0.9를 곱합니다.
모든 예측을 단계별로 진행할 수 있습니다. 실제로 발생한 결과와 관련된 확률을 찾을 때마다 이전 결과와 곱할 수 있습니다. 이것이 우도 값입니다.

<h2>From Likelihood to Log Loss</h2>

Each prediction is between 0 and 1. If you multiply enough numbers in this range, the result gets so small that computers can't keep track of it. So, as a clever computational trick, we instead keep track of the log of the Likelihood. This is in a range that's easy to keep track of. We multiply this by negative 1 to maintain a common convention that lower loss scores are better.

각 예측은 0과 1 사이입니다.이 범위에 충분한 수를 곱하면 결과가 너무 작아서 컴퓨터가 이를 추적 할 수 없습니다. 따라서 현명한 계산 기법으로 우도 값의 로그를 추적합니다. 추적하기 쉬운 범위에 있습니다. 손실 점수가 낮을수록 일반적인 관례를 유지하기 위해 -1을 곱합니다.

# Reference.
https://www.kaggle.com/dansbecker/what-is-log-loss