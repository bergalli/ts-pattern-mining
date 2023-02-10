
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import optimize_a_binomial_model, retrain_best_binomial_model, extract_glm_coefficients


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                optimize_a_binomial_model,
                inputs=dict(
                    data_store="data_store",
                    model_perf_metric="params:model_perf_metric",
                    n_folds="params:n_folds",
                    experiment_name="experiment_name",
                    experiment_tags="experiment_tags",
                    run_uuid="run_uuid",
                    population_size="params:genetic_optim.population_size",
                    population_decay="params:genetic_optim.population_decay",
                    max_generation="params:genetic_optim.max_generation",
                    endog_test="endog_test",
                    exog_test="exog_test",
                ),
                outputs="optimization_results",
            ),
            node(
                retrain_best_binomial_model,
                inputs=dict(
                    data_store="data_store",
                    optimization_results="optimization_results",
                    exog_test="exog_test",
                    endog_test="endog_test",
                    model_perf_metric="params:model_perf_metric",
                ),
                outputs="creative_score_model",
            ),
            node(
                extract_glm_coefficients,
                inputs=dict(
                    dataprep_table="df_input",
                    trained_model="creative_score_model",
                ),
                outputs="coefs_frame",
            ),
        ]
    )
