import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader  


class PairDataset(Dataset):
    """One '(question, relevant document)' pair per question of a corpus split."""

    def __init__(self, split, random_positive: bool):
        self.split = split  #a corpus split (corpus.train or corpus.val)
        self.random_positive = random_positive  #True -> pick a random relevant document each time (training) | False -> always pick the first relevant document (validation)

    def __len__(self):
        # One item per question
        return len(self.split)  

    def __getitem__(self, idx):
        row = self.split[idx]
        question = row['question']     # the question text
        documents = row['text']                # the list of relevant document texts

        if self.random_positive:    
            document = random.choice(documents)            # one document chosen at random
        else:
            document = documents[0]         # always the first document

        return question, document


class PairCollator:

    """ Turns a list of (question, document) pairs into a batch of tokenized tensors. """

    def __init__(self, tokenizer, max_query_length: int, max_document_length: int):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length

    def __call__(self, batch):

        # batch: list of B (question, document) pairs from PairDataset
        questions = [q for q, d in batch]      # list of B question texts
        documents = [d for q, d in batch]      # list of B document texts

        q_tokens = self.tokenizer(questions, padding=True, truncation=True,
                                  max_length=self.max_query_length, return_tensors="pt")
        d_tokens = self.tokenizer(documents, padding = True, truncation = True, max_length=self.max_document_length, return_tensors = 'pt')

        return {
            "q_input_ids": q_tokens["input_ids"],
            "q_attention_mask": q_tokens["attention_mask"],
            "d_input_ids": d_tokens["input_ids"],
            "d_attention_mask": d_tokens["attention_mask"],
        }


def build_dataloaders(corpus, tokenizer, batch_size: int, max_query_length: int, max_document_length: int):

    """Build the training and validation DataLoaders of (question, document) pairs."""

    collate_fn = PairCollator(tokenizer, max_query_length, max_document_length)

    train_dataset = PairDataset(corpus.train, random_positive=True)
    val_dataset = PairDataset(corpus.val, random_positive=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

    return train_loader, val_loader