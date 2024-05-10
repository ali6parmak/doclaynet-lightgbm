import pickle
from os import listdir
from os.path import join
from pathlib import Path
from time import time
from tqdm import tqdm
from pdf_features.PdfFeatures import PdfFeatures
from pdf_tokens_type_trainer.ModelConfiguration import ModelConfiguration
from pdf_tokens_type_trainer.TokenTypeTrainer import TokenTypeTrainer


BENCHMARK_MODEL = join(Path(__file__).parent, "model", "token_type.model")
model_configuration = ModelConfiguration()

doclaynet_data_base_path: str = str(Path(__file__).parent.parent)


def get_pdf_features(split: str):
    cached_data_path: str = join(doclaynet_data_base_path, "cached_data_pdf_features", split)
    files_to_skip_path: str = join(doclaynet_data_base_path, f"pdfs_to_check_{split}")
    files_to_skip = [file.replace(".pdf", ".pickle") for file in listdir(files_to_skip_path)]
    labeled_data: list[PdfFeatures] = []
    file_names = [file_name for file_name in sorted(listdir(cached_data_path)) if file_name not in files_to_skip]
    print(f"PdfFeatures loading for [{split}] data: ")
    for file in tqdm(file_names):
        file_path = join(cached_data_path, file)
        with open(file_path, "rb") as pickle_file:
            pdf_features: PdfFeatures = pickle.load(pickle_file)
        labeled_data.append(pdf_features)
    return labeled_data


def train_for_benchmark():
    Path(BENCHMARK_MODEL).parent.mkdir(exist_ok=True)
    train_pdf_features = get_pdf_features("train")
    print(f"Total num of training PDFs: {len(train_pdf_features)}")
    trainer = TokenTypeTrainer(train_pdf_features, model_configuration)
    labels = [token.token_type.get_index() for token in trainer.loop_tokens()]
    trainer.train(BENCHMARK_MODEL, labels)


if __name__ == "__main__":
    print("start")
    start = time()
    train_for_benchmark()
    print("finished in", time() - start, "seconds")
