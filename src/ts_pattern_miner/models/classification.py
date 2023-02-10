import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ts_pattern_miner.models.base import RayModelOptimizer


class ClassificationStatsmodels(RayModelOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.endog_test, self.exog_test = None, None

    def _fit(self, endog, exog, freq_weights=None, **fit_params):
        if freq_weights is not None:
            freq_weights = freq_weights.values.ravel().astype(float)
        self.model = self.new_model(endog=endog.values, exog=exog, freq_weights=freq_weights)
        self.model = self.model.fit(**fit_params)
        return self

    def predict(self, exog, **kwargs):
        preds = self.model.predict(exog)
        return preds

    def _score(self, endog, exog, metric, **kwargs):
        """
        - f1 score:
            Le F1-score est une métrique de classification qui mesure la capacité d’un modèle à bien prédire les individus positifs, tant en termes de precision (taux de prédictions positives correctes) qu’en termes de recall (taux de positifs correctement prédits).

        :param from_freeze:
        :param endog:
        :param exog:
        :param metric:
        :return: score to maximize
        """
        y_true = endog >= 0.5
        y_pred_proba = self.predict(exog)
        y_pred_bool = y_pred_proba >= 0.5
        if metric == "acc":
            return accuracy_score(y_true, y_pred_bool)
        elif metric == "f1":
            return f1_score(y_true, y_pred_bool)
        elif metric == "auc":
            return roc_auc_score(y_true, y_pred_proba)
        else:
            raise ValueError

    def get_coefficients(self):
        model_summary = self.model.summary()
        coefficients = pd.DataFrame(
            model_summary.tables[1].data[1:],
            columns=["variable"] + model_summary.tables[1].data[0][1:],
        )
        coefficients.loc[:, ["coef", "std err", "z", "P>|z|", "[0.025", "0.975]"]] = coefficients[
            ["coef", "std err", "z", "P>|z|", "[0.025", "0.975]"]
        ].astype(float)
        coefficients = coefficients.rename(columns={"P>|z|": "pvalue"})
        return coefficients
