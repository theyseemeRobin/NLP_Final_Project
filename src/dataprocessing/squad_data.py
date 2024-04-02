import pandas as pd
import torch
from torch.utils.data import Dataset

from my_util import fetch_device


def load_data(json_path):
    """Load SQuAD data from a given JSON file into a pandas DataFrame."""
    data = pd.read_json(json_path)
    rows = []
    for item in data['data']:
        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                answers = {"text" : [], "answer_start" : []}
                for a in qa["answers"]:
                    answers["text"].append(a["text"])
                    answers["answer_start"].append(a["answer_start"])
                row = {
                    'id': qa['id'],
                    'title': item['title'],
                    'context': paragraph['context'],
                    'question': qa['question'],
                    'answers': answers,

                }
                rows.append(row)
    return pd.DataFrame(rows)


class SquadData(Dataset):
    def __init__(self, json_path, tokenizer, max_tokens=256, stride=50, device=fetch_device()):
        """
        Parameters
        ----------
        json_path : str
            Path to json file
        tokenizer : transformers.PreTrainedTokenizerFast
        max_tokens : int
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = device
        self.json_path = json_path
        self.data = load_data(json_path=self.json_path)

        self.stride = stride
        self.processed_data  = self.preprocess()

    def preprocess(self):
        questions = [q.strip() for q in self.data["question"].tolist()]
        processed_data = self.tokenizer(
            questions,
            self.data["context"].tolist(),
            max_length=self.max_tokens,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = processed_data["offset_mapping"]
        sample_map = processed_data["overflow_to_sample_mapping"]
        answers = self.data["answers"].tolist()
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            # If there is no answer
            if len(answers[sample_idx]["text"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = processed_data.sequence_ids(i)

            # Find the start and end of the context (SQUAD v2 might have such questions)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        processed_data["start_positions"] = start_positions
        processed_data["end_positions"] = end_positions
        return processed_data

    def __getitem__(self, idx):
        """
        For a given datapoint `idx`, return the sequence of token ids (input_ids), the attention mask, the start
        position of the answer and the end position of the answer as tensors.
        """
        return {
            'input_ids': torch.tensor(self.processed_data['input_ids'][idx], dtype=torch.long, device=self.device),
            'attention_mask': torch.tensor(self.processed_data['attention_mask'][idx], dtype=torch.long,
                                           device=self.device),
            'start_positions': torch.tensor(self.processed_data["start_positions"][idx], dtype=torch.long,
                                            device=self.device),
            'end_positions': torch.tensor(self.processed_data["end_positions"][idx], dtype=torch.long,
                                          device=self.device)
        }

    def __len__(self):
        return len(self.processed_data["attention_mask"])


