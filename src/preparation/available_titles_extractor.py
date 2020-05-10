import gzip
from data_structure.click_stream_pre_processed import ClickStreamPreProcessed
from utils.utils import clean_title

class AvailableTitlesExtractor:
    def __init__(
        self,
        wiki_titles_path="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/source/enwiki-20200401-all-titles-in-ns0.gz",
    ):
        with gzip.open(wiki_titles_path, mode="rt",) as f:
            self._wiki_articles_titles = list(map(clean_title, f.read().splitlines()))

        self._click_stream_titles = ClickStreamPreProcessed().get_titles()

    def run(
        self,
        save_path="/Users/dnascimentodepau/Documents/python/thesis/thesis-davi/data/processed/available_titles.txt",
    ):
        available_titles = list(set(self._wiki_articles_titles).intersection(self._click_stream_titles))

        print("Unique Wiki articles {}".format(len(self._wiki_articles_titles)))
        print("Unique Click Stream titles {}".format(len(self._click_stream_titles)))
        print("Final unique titles {}".format(len(available_titles)))

        with open(save_path, "w") as output:
            output.write("\n".join(available_titles))


if __name__ == "__main__":
    gen = AvailableTitlesExtractor()
    gen.run()
