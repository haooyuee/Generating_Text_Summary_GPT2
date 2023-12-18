import pandas as pd
import pytorch_lightning as pl
import torch
import tqdm
from rouge import Rouge
from bert_score import score
from transformers import logging
from model import MyT5Model

SAVE_NAME = "T5_14"
LOAD_NAME = "T5_14"

TRAINING_FLAG = True

EARLY_STOPPING_PATIENCE_EPOCHS: int = 2
BATCH_SIZE: int = 1
MODEL_NAME: str = "t5-base"
LR: float = 0.0001
SCHEDULER_LR: bool = True

# predict parameters
MAX_LENGTH: int = 100
NUM_BEAMS: int = 4
TOP_K: int = 50
TOP_P: float = 0.95
REPETITION_PENALTY: float = 2.5
LENGTH_PENALTY: float = 0.6


def data_prepare(datasets_list):
    processed_datasets_list = []

    for df in datasets_list:
        df = df.rename(columns={"summary": "target_text", "document": "source_text"})
        df = df[['source_text', 'target_text']]

        df['source_text'] = "summarize: " + df['source_text']
        processed_datasets_list.append(df)

    print('All converted training samples: \n', processed_datasets_list[0])

    for df in processed_datasets_list:
        print(df.shape)

    return processed_datasets_list


if __name__ == '__main__':

    torch.cuda.empty_cache()
    pl.seed_everything(42)
    train_data_path = "train_dataset.csv"
    valid_data_path = "valid_dataset.csv"
    test_data_path = "test_dataset.csv"

    t5_train = pd.read_csv(train_data_path, sep='\t')
    t5_valid = pd.read_csv(valid_data_path, sep='\t')
    t5_test = pd.read_csv(test_data_path, sep='\t')

    df_list = data_prepare([t5_train, t5_valid, t5_test])
    t5_train_df, t5_valid_df, t5_test_df = df_list[0], df_list[1], df_list[2]

    if TRAINING_FLAG:
        t5_model = MyT5Model(SAVE_NAME)
        t5_model.from_pretrained(model_name=MODEL_NAME)
        t5_model.train(
            train_df=t5_train_df,
            eval_df=t5_valid_df,
            source_max_token_len=768,
            target_max_token_len=64,
            batch_size=BATCH_SIZE,
            max_epochs=10,
            use_gpu=True,
            early_stopping_patience_epochs=EARLY_STOPPING_PATIENCE_EPOCHS
        )
        print('Model Trained!')
    else:
        t5_model = None

    if TRAINING_FLAG:
        test_t5_model = t5_model
    else:
        test_t5_model = MyT5Model()
        test_t5_model.load_model("outputs/" + LOAD_NAME, use_gpu=True)
    print('Model Loaded!')

    logging.set_verbosity_error()

    my_rouge = Rouge()
    rouge_1, rouge_2, rouge_l_f1 = 0, 0, 0
    Bert_P_sum, Bert_R_sum, Bert_F1_sum = 0, 0, 0
    output_text_list, label_text_list = [], []

    for ind in tqdm.tqdm(range(len(t5_test_df))):
        input_text = t5_test_df.iloc[ind]['source_text']
        output_text = test_t5_model.predict(input_text,
                                            max_length=MAX_LENGTH,
                                            num_beams=NUM_BEAMS,
                                            top_k=TOP_K,
                                            top_p=TOP_P,
                                            repetition_penalty=REPETITION_PENALTY,
                                            length_penalty=LENGTH_PENALTY)
        label_text = t5_test_df.iloc[ind]['target_text']

        output_text_list.append(output_text[0])
        label_text_list.append(label_text)

        result = my_rouge.get_scores(output_text, [label_text], avg=True)
        rouge_1 += result['rouge-1']['f']
        rouge_2 += result['rouge-2']['f']
        rouge_l_f1 += result['rouge-l']['f']

    print('Average ROUGE Scores on Test Set: Rouge_1: {}, Rouge_2: {}, Rouge_l_f1: {}'.format(
        rouge_1 / len(t5_test_df), rouge_2 / len(t5_test_df), rouge_l_f1 / len(t5_test_df)))

    P, R, F1 = tqdm.tqdm(score(output_text_list, label_text_list, lang="en"))
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    print('Average BERT_Scores on Test Set: BERTScore_Precision: {}, BERTScore_Recall: {}, BERTScore_F1: {}'.format(
        avg_precision, avg_recall, avg_f1))
