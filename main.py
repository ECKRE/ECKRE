import argparse
import os
from trainer import Trainer
from utils import init_logger, load_tokenizer, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples


def main(args):
    '''
    the main process of SRGLHRE
    '''
    init_logger()
    tokenizer = load_tokenizer(args)
    train_datasets = load_and_cache_examples(args, tokenizer, mode="train")
    test_datasets = load_and_cache_examples(args, tokenizer, mode="test")
    trainer = Trainer(args, train_datasets=train_datasets, test_datasets=test_datasets)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    parser.add_argument("--model_type", default="albert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--data_dir", default="./data/semeval", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default="./albert_model/albert-xxlarge-v1", help="Path to input model")
    parser.add_argument("--model_save_dir", default="./model", type=str, help="Path to save model")
    parser.add_argument("--alpha", default=0.3, type=int, help="the weight coefficient of classification loss")
    parser.add_argument("--beta", default=0.3, type=int, help="the corrected coefficient")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")


    args = parser.parse_args()

    main(args)
