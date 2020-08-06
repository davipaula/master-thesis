import logging

import torch
from tqdm import tqdm
import pandas as pd

from modeling.wikipedia2vec_model import Wikipedia2VecModel
from utils.constants import CLICK_STREAM_TEST_DATASET_PATH, RESULT_FILE_COLUMNS_NAMES
from utils.utils import cosine_similarity

logger = logging.getLogger(__name__)

LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

model_name = "wikipedia2vec_cosine"


def get_top_n_cosine():
    batch_size = 32

    click_stream_test_dataset = torch.load(CLICK_STREAM_TEST_DATASET_PATH)
    test_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
    }

    logger.info("Opening test dataset")
    click_stream_test = torch.utils.data.DataLoader(
        click_stream_test_dataset, **test_params
    )

    logger.info("Initializing model")
    model = Wikipedia2VecModel()

    predictions_list = pd.DataFrame(columns=RESULT_FILE_COLUMNS_NAMES)

    for row in tqdm(click_stream_test):
        source_article_vector = model.get_entity_vector(row["source_article"])
        target_article_vector = model.get_entity_vector(row["target_article"])

        cosine_similarities = [
            cosine_similarity(source_article_vector[idx], target_article_vector[idx])
            for idx in range(len(source_article_vector))
        ]

        batch_results = pd.DataFrame(
            zip(
                [model_name] * batch_size,
                row["source_article"],
                row["target_article"],
                row["click_rate"].tolist(),
                cosine_similarities,
            ),
            columns=RESULT_FILE_COLUMNS_NAMES,
        )

        predictions_list = predictions_list.append(batch_results, ignore_index=True)

    predictions_list.to_csv(
        f"./results/test/results_{model_name}_level_test.csv", index=False
    )
    logger.info(f"Model {model_name}. Evaluation finished.")


if __name__ == "__main__":
    get_top_n_cosine()
