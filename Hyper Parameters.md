XGBoost Parameters

    - n_estimators
    약한 학습기의 개수(반복 수행 횟수)
    
    - learning_rate
    부스팅 스탭을 반복적으로 수행할 때 업데이트 되는 학습률 값
    매 부스팅 스탭마다 weight를 주어 부스팅 과정에서 과적합이 일어나지 않도록 함
    range: [0,1]

    - gamma
    트리의 리프 노드를 추가적으로 나눌지 결정할 최소 손실 감소 값
    해당 값보다 큰 손실(loss)이 감소된 경우에 리프 노드를 분리함. 값이 클수록 과적합 감소 효과가 있음
    range: [0,∞]

    - subsample
    트리가 커져 과적합 되는 것을 제어하기 위해 데이터를 샘플링하는 비율 지정
    일반적으로 0.5 ~ 1 사이 값을 지정. (0.5:전체 데이터의 절반을 트리로 생성)
    range: [0,1]

    - max_depth
    트리의 최대 깊이
    range: [0,∞]

    - colsample_bytree
    트리 생성에 필요한 feature를 임의로 sampling 하는데 사용됨
    많은 feature가 있는 경우 과적합 조정하는 데 적용
    range: [0, 1]

    - objective
    loss function 함수 종류

    - early_stopping_rounds
    더 이상 비용 평가 지표가 감소하지 않는 최대 반복횟수
    
    - eval_set
    평가를 수행하는 별도의 검증 데이터 세트
    일반적으로 검증 데이터 세트에서 반복적으로 비용 감소 성능 평가

    - eval_metric
    반복 수행 시 사용하는 비용 평가 지표
    classifier에서는 'error'를 default로 사용

    1. 높은  learning rate를 선택하고, cv 를 사용해서 최적의 트리 수를 찾음
    2. 트리의 파라미터(max_depth, min_child_weight, gmma, subsample, solsample_bytree)를 결정
    3. 정규화 파라미터(lambda, alpha)를 조정해서 모델의 복잡도를 감소시킴
    4. Learning rate를 더 낮추고 최적의 파라미터 결정


LightGBM Parameters

    - num_iterations
    boosting 반복량
    range : ∞, default = 100

    - learning_rate
    부스팅 스탭을 반복적으로 수행할 때 업데이트 되는 학습률 값
    매 부스팅 스탭마다 weight를 주어 부스팅 과정에서 과적합이 일어나지 않도록 함
    range: [0,1]

    - max_depth
    트리의 최대 깊이
    range: ∞, default = -1
    
    - num_leaves
    최대 리프 노드 개수
    range: ∞, default = 31

    - bagging_fraction
    트리가 커져 과적합 되는 것을 제어하기 위해 데이터를 샘플링하는 비율(bagging의 비율) 지정
    일반적으로 0.5 ~ 1 사이 값을 지정. (0.5:전체 데이터의 절반을 트리로 생성)
    range: ∞, default = 1.0
    
    - feature_fraction
    트리 생성에 필요한 feature를 임의로 sampling 하는데 사용됨
    많은 feature가 있는 경우 과적합 조정하는 데 적용
    range: ∞, default = 1.0
    
    - early_stopping_rounds
    학습 조기 종료를 위한 early stopping interval 값

    - objective
    loss function 함수 종류
    
    - eval_set
    평가를 수행하는 별도의 검증 데이터 세트
    일반적으로 검증 데이터 세트에서 반복적으로 비용 감소 성능 평가

    - eval_metric
    반복 수행 시 사용하는 비용 평가 지표
    classifier에서는 'error'를 default로 사용

    ref.
    https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#tune-parameters-for-the-leaf-wise-best-first-tree


CatBoost Parameters

    - iterations
    boosting 반복량

    - learning rate
    부스팅 스탭을 반복적으로 수행할 때 업데이트 되는 학습률 값
    매 부스팅 스탭마다 weight를 주어 부스팅 과정에서 과적합이 일어나지 않도록 함
    
    - early_stopping_rounds
    학습 조기 종료를 위한 early stopping interval 값

    - depth
    트리의 깊이
    대부분의 경우 최적의 깊이 범위는 4-10. 6-10 범위의 값을 권장함

    - l2_leaf_reg
    비용 함수의 L2 정규화 계수 값.

    - loss_function
    단일 분류는 'LogLoss' or 'CrossEntropy' 사용. 멀티 분류는 'MultiClass' 사용.

    ref.
    https://towardsdatascience.com/an-example-of-hyperparameter-optimization-on-xgboost-lightgbm-and-catboost-using-hyperopt-12bc41a271e

LogisticRegression Parameters

    - C
    규칙의 강도를 조절하는 alpha 값의 역수. C = 1/alpha. C값이 작을수록 규제 강도가 큼
    default = 1.0

    - penalty
    페널티를 부여할 때 규제의 유형을 설정
    value = ('l1', 'l2', 'elasticnet', 'none'), default='l2'

    - dual
    Dual Formulation 인지 Primal Formulation 인지를 결정 (True, False)
    default = False

    - tol
    지정한 값만큼 감소되지 않으면 반복을 중단하도록 하는 값
    default = 1e-4

    - fit_intercept
    의사 결정 기능에 상수를 추가할 지 여부 결정 (True, False)
    default = True

    - intercept_scaling
    synthetic feature weight(합성 피쳐 가중치)는 다른 모든 피처와 마찬가지로 l1, l2 정규화를 따름. 
    synthetic feature weight (따라서 intercept 에서)에 대한 정규화의 영향을 줄이려면 intercept_scaling을 증가시켜야합니다.
    default = 1

    - class_weight
    클래스에 대한 가중치들의 값
    작은 데이터 세트의 경우 'liblinear'가 좋은 선택이지만 'sag'및 'saga'는 큰 데이터의 경우 더 빠름
    value = (dict or 'balanced'), default = None

    - solver
    최적화에 사용할 알고리즘 결정 
    value = ('newton-cg', 'lbfgs', liblinear', 'sag', 'saga'),  default = 'lbfgs'

    - max_iter
    solver가 수렴하게 만드는 최대 반복 횟수 값
    default = 100

    - multi_class
    선택한 옵션이 'ovr'이면 binary problem을 각 label에 fit함. solver = 'liblinear' 인 경우 'multinomial'을 사용할 수 없음
    'auto'은 데이터가 binary이거나 solver = 'liblinear' 인 경우 'ovr'를 선택하고 그렇지 않으면 'multinomial'을 선택함
    value = ('auto', 'ovr', 'multinomial'), default='auto'

    - warm_start
    이전 호출에 사용했던 solution 을 재사용 할지 여부 결정 
    value = (True, False), default = False

    - n_jobs
    병렬 처리 시 사용 할 CPU 코어의 수

        