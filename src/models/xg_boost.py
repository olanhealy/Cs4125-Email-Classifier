from xgboost import XGBClassifier

class XGBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=0):
        self.model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
