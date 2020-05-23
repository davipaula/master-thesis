import pandas as pd

BASE_RESULTS_PATH = "./results/test/"
DOC2VEC_RESULTS_PATH = BASE_RESULTS_PATH + "results_doc2vec_level_test.csv"
WIKIPEDIA2VEC_RESULTS_PATH = BASE_RESULTS_PATH + "results_wikipedia2vec_level_test.csv"
SMASH_RNN_WORD_LEVEL_RESULTS_PATH = BASE_RESULTS_PATH + "results_word_level.csv"
SMASH_RNN_SENTENCE_LEVEL_RESULTS_PATH = BASE_RESULTS_PATH + "results_sentence_level.csv"
SMASH_RNN_PARAGRAPH_LEVEL_RESULTS_PATH = BASE_RESULTS_PATH + "results_paragraph_level.csv"

# For debugging purposes only
BASE_VALIDATION_RESULTS_PATH = "./results/"
SMASH_RNN_WORD_LEVEL_VALIDATION_RESULTS_PATH = BASE_VALIDATION_RESULTS_PATH + "results_word_level_validation.csv"
SMASH_RNN_SENTENCE_LEVEL_VALIDATION_RESULTS_PATH = (
    BASE_VALIDATION_RESULTS_PATH + "results_sentence_level_validation.csv"
)
SMASH_RNN_PARAGRAPH_LEVEL_VALIDATION_RESULTS_PATH = (
    BASE_VALIDATION_RESULTS_PATH + "results_paragraph_level_validation.csv"
)


class ResultsAnalyzer:
    def __init__(self):
        self.results = self.build_models_results()

        self.top_articles = self.build_top_n_matrix_by_article()

    def build_models_results(self):
        __models = {
            "doc2vec": DOC2VEC_RESULTS_PATH,
            "wikipedia2vec": WIKIPEDIA2VEC_RESULTS_PATH,
            "smash_rnn_word_level": SMASH_RNN_WORD_LEVEL_RESULTS_PATH,
            "smash_rnn_sentence_level": SMASH_RNN_SENTENCE_LEVEL_RESULTS_PATH,
            "smash_rnn_paragraph_level": SMASH_RNN_PARAGRAPH_LEVEL_RESULTS_PATH,
        }

        columns_names = [
            "model",
            "source_article",
            "target_article",
            "actual_click_rate",
            "predicted_click_rate",
        ]

        results = pd.DataFrame(columns=columns_names)

        for model_path in __models.values():
            results = results.append(pd.read_csv(model_path))

        return results

    def get_top_n_predicted_by_article_and_model(self, source_article: str, model: str, n=5):
        model_results = self.results[
            (self.results["model"] == model) & (self.results["source_article"] == source_article)
        ]
        model_results = (
            model_results.sort_values("predicted_click_rate", ascending=False).groupby("source_article").head(n)
        )

        model_results["is_in_top_10"] = False
        actual_top_articles = self.top_articles[self.top_articles["source_article"] == source_article][
            "target_article"
        ].unique()
        model_results.loc[model_results["target_article"].isin(actual_top_articles), "is_in_top_10"] = True

        return model_results

    def build_top_n_matrix_by_article(self, n=5):
        actual_results = (
            self.results[self.results["model"] == "paragraph"]
            .sort_values(by=["source_article", "actual_click_rate"], ascending=[True, False])
            .groupby("source_article")
            .head(n)
            .drop(["actual_click_rate", "predicted_click_rate"], axis=1)
        )

        actual_results["model"] = "actual click rate"

        return actual_results

    def build_validation_models_results(self):
        # For debugging purposes only
        columns_names_validation = [
            "source_article",
            "target_article",
            "actual_click_rate",
            "predicted_click_rate",
        ]
        __models_validation = {
            "smash_rnn_word_level": SMASH_RNN_WORD_LEVEL_VALIDATION_RESULTS_PATH,
            "smash_rnn_sentence_level": SMASH_RNN_SENTENCE_LEVEL_VALIDATION_RESULTS_PATH,
            "smash_rnn_paragraph_level": SMASH_RNN_PARAGRAPH_LEVEL_VALIDATION_RESULTS_PATH,
        }
        results_validation = pd.DataFrame(columns=columns_names_validation)
        for model_path in __models_validation.values():
            results_validation = results_validation.append(pd.read_csv(model_path))

        return results_validation

    def run(self):
        source_article = "Kirsten Dunst"
        model_names = ["doc2vec", "wikipedia2vec", "paragraph", "word"]

        for model_name in model_names:
            predicted = self.get_top_n_predicted_by_article_and_model(source_article, model_name)

            print(predicted)

        actual = self.build_top_n_matrix_by_article()
        print(actual)


if __name__ == "__main__":
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    results = ResultsAnalyzer()
    print(results.get_top_n_predicted_by_article_and_model("Lil Wayne", "word"))
    print(results.get_top_n_predicted_by_article_and_model("Lil Wayne", "paragraph"))
