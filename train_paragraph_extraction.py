import os
import pickle
from os import listdir
from os.path import join
from pathlib import Path
from time import time
from tqdm import tqdm
from paragraph_extraction_trainer.PdfParagraphTokens import PdfParagraphTokens
from paragraph_extraction_trainer.ParagraphExtractorTrainer import ParagraphExtractorTrainer
from paragraph_extraction_trainer.model_configuration import MODEL_CONFIGURATION

BENCHMARK_MODEL_PATH = Path(join(Path(__file__).parent, "model", "paragraph_extraction.model"))
doclaynet_data_base_path: str = str(Path(__file__).parent.parent)


def loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list: list[PdfParagraphTokens]):
    for pdf_paragraph_tokens in pdf_paragraph_tokens_list:
        for page in pdf_paragraph_tokens.pdf_features.pages:
            if not page.tokens:
                continue
            for token, next_token in zip(page.tokens, page.tokens[1:]):
                yield pdf_paragraph_tokens, token, next_token
            yield pdf_paragraph_tokens, page.tokens[-1], page.tokens[-1]


def get_paragraph_tokens(split: str):
    cached_data_path: str = join(doclaynet_data_base_path, "cached_data", split)
    files_to_skip_path: str = join(doclaynet_data_base_path, f"pdfs_to_check_{split}")
    files_to_skip = [file.replace(".pdf", ".pickle") for file in os.listdir(files_to_skip_path)]
    labeled_data: list[PdfParagraphTokens] = []
    file_names = [file_name for file_name in sorted(listdir(cached_data_path)) if file_name not in files_to_skip]
    print(f"PdfParagraphTokens loading for [{split}] data: ")
    for file in tqdm(file_names):
        file_path = join(cached_data_path, file)
        with open(file_path, "rb") as pickle_file:
            pdf_paragraph_tokens: PdfParagraphTokens = pickle.load(pickle_file)
        labeled_data.append(pdf_paragraph_tokens)
    return labeled_data


def train_for_benchmark():
    pdf_paragraph_tokens_list = get_paragraph_tokens("train")
    print("Total num of training PDFs:", len(pdf_paragraph_tokens_list))
    pdf_features_list = [pdf_paragraph_tokens.pdf_features for pdf_paragraph_tokens in pdf_paragraph_tokens_list]
    trainer = ParagraphExtractorTrainer(pdfs_features=pdf_features_list, model_configuration=MODEL_CONFIGURATION)

    labels = []
    for pdf_paragraph_tokens, token, next_token in loop_pdf_paragraph_tokens(pdf_paragraph_tokens_list):
        labels.append(pdf_paragraph_tokens.check_same_paragraph(token, next_token))
    os.makedirs(BENCHMARK_MODEL_PATH.parent, exist_ok=True)
    trainer.train(str(BENCHMARK_MODEL_PATH), labels)


if __name__ == "__main__":
    print("start")
    start = time()
    train_for_benchmark()
    print("finished in", int(time() - start), "seconds")
