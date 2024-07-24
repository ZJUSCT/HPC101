import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import math
import warnings
import multiprocessing
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tqdm import tqdm

# os.environ['HTTP_PROXY'] = 'http://172.25.2.253:7890'
# os.environ['HTTPS_PROXY'] = 'https://172.25.2.253:7890'

datasetDir = "/home/huangjinjun/workspace/testdataset"

dataset = load_dataset(path=datasetDir)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(batch):
    input_texts = [data['zh'] for data in batch['translation']]
    target_texts = [data['en'] for data in batch['translation']]
    
    inputs = tokenizer(input_texts, padding='max_length', truncation=True, max_length=128)
    targets = tokenizer(target_texts, padding='max_length', truncation=True, max_length=128)
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': targets['input_ids'],
        'tgt_embedding_mask': targets['attention_mask'],
    }
    
tokenized_dataset = {}
timestamp_set = []
timecnt = 0
# train_dataset =  dataset['train'].map(preprocess_function, batched=True, remove_columns=["translation"])
# val_dataset =  dataset['validation'].map(preprocess_function, batched=True, remove_columns=["translation"])
def getNowTimeInterval():
    global timestamp_set, timecnt
    timestamp_set.append(time.perf_counter())
    timecnt += 1
    return timestamp_set[timecnt-1] - timestamp_set[timecnt-2]

def preprocess_and_save(dataset, file_prefix, save_dir):
    print(f'Preprocessing and saving {file_prefix} datasets...')
    input_ids_path = os.path.join(save_dir, f"{file_prefix}_input_ids.pt")
    attention_mask_path = os.path.join(save_dir, f"{file_prefix}_attention_mask.pt")
    labels_path = os.path.join(save_dir, f"{file_prefix}_labels.pt")
    tgt_embedding_mask_path = os.path.join(save_dir, f"{file_prefix}_tgt_embedding_mask.pt")
    global timestamp_set, timecnt, tokenized_dataset
    
    if not (os.path.exists(input_ids_path) and 
            os.path.exists(attention_mask_path) and 
            os.path.exists(labels_path) and
            os.path.exists(tgt_embedding_mask_path)):
        tokenized_dataset = dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=["translation"]
        )
        print(tokenized_dataset)
        print(f"------{file_prefix} dataset Tokenize time: {getNowTimeInterval():.2f} seconds")    
        torch.save(tokenized_dataset['input_ids'], input_ids_path)
        print(f"------Tensor input_ids save time: {getNowTimeInterval():.2f} seconds")    
        torch.save(tokenized_dataset['attention_mask'], attention_mask_path)
        print(f"------Tensor attention_mask save time: {getNowTimeInterval():.2f} seconds")    
        torch.save(tokenized_dataset['labels'], labels_path)
        print(f"------Tensor labels save time: {getNowTimeInterval():.2f} seconds")    
        torch.save(tokenized_dataset['tgt_embedding_mask'], tgt_embedding_mask_path)
        print(f"------Tensor tgt_embedding_mask save time: {getNowTimeInterval():.2f} seconds")    
        
print("Preprocessing Started!")
start = time.perf_counter()
timestamp_set.append(start)
timecnt += 1

preprocess_and_save(dataset['train'], 'train', datasetDir)
preprocess_and_save(dataset['validation'], 'validation', datasetDir)

def preprocess_chunk(chunk):
    input_ids, attention_mask, labels, tgt_embedding_mask = chunk
    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'labels': torch.tensor(labels),
        'tgt_embedding_mask': torch.tensor(tgt_embedding_mask)
    }

def load_preprocessed_data(file_prefix, save_dir) -> Dataset:
    print(f'Loading {file_prefix} datasets...')
    
    global timestamp_set, timecnt, tokenized_dataset
    
    input_ids = torch.load(os.path.join(save_dir, f"{file_prefix}_input_ids.pt"))
    print(f"------Tensor input_ids load time: {getNowTimeInterval():.2f} seconds")    
    attention_mask = torch.load(os.path.join(save_dir, f"{file_prefix}_attention_mask.pt"))
    print(f"------Tensor attention_mask load time: {getNowTimeInterval():.2f} seconds")    
    labels = torch.load(os.path.join(save_dir, f"{file_prefix}_labels.pt"))
    print(f"------Tensor labels load time: {getNowTimeInterval():.2f} seconds")    
    tgt_embedding_mask = torch.load(os.path.join(save_dir, f"{file_prefix}_tgt_embedding_mask.pt"))
    print(f"------Tensor tgt_embedding_mask load time: {getNowTimeInterval():.2f} seconds")    
    
    
    # Create the Dataset object with multiprocessing 
    # features = Features({
    #     'input_ids': Sequence(feature=Value(dtype='int64')),
    #     'attention_mask': Sequence(feature=Value(dtype='int64')),
    #     'labels': Sequence(feature=Value(dtype='int64')),
    #     'tgt_embedding_mask': Sequence(feature=Value(dtype='int64')),
    # })
    
    # chunk_size = 20000
    # chunks = [(input_ids[i:i + chunk_size], attention_mask[i:i + chunk_size], labels[i:i + chunk_size], tgt_embedding_mask[i:i + chunk_size])
    #       for i in range(0, len(input_ids), chunk_size)]
    # with multiprocessing.Pool() as pool:
    #     processed_chunks = pool.map(preprocess_chunk, chunks);
    # final_data = {key: [] for key in processed_chunks[0].keys()}
    # for chunk in processed_chunks:
    #     for key in final_data.keys():
    #         final_data[key].extend(chunk[key])

    
    # dataset = Dataset.from_dict(final_data, features=features)
    # print(f"------Dataset created time: {getNowTimeInterval():.2f} seconds")   
    
    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(feature=Value(dtype='int64')),
        'labels': Sequence(feature=Value(dtype='int64')),
        'tgt_embedding_mask': Sequence(feature=Value(dtype='int64')),
    })

    # Create the Dataset object
    
    dataset = Dataset.from_dict({
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'labels': torch.tensor(labels),
        'tgt_embedding_mask': torch.tensor(tgt_embedding_mask)
    }, features=features)
    print(f"------Dataset created time: {getNowTimeInterval():.2f} seconds")    
    
    return dataset

# if(not tokenized_dataset):
val_dataset = load_preprocessed_data('validation', datasetDir)
train_dataset = load_preprocessed_data('train', datasetDir)

warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(max_seq_length, dropout=0)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask, tgt_mask, tgt_key_padding_mask):
        src = self.embedding(src)  # batch_size x seq_length x d_model
        tgt = self.embedding(tgt)  # batch_size x seq_length x d_model
        
        src = self.pos_encoder(src) 
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.fc(output) #  batch_size x seq_length x vocab_size

        return output
    
class CustomDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        labels = torch.tensor(item['labels'])
        tgt_embedding_mask = torch.tensor(item['tgt_embedding_mask'])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'tgt_embedding_mask': tgt_embedding_mask
        }

def train(model, dataset, epochs, batch_size, lr):
    print(f'Training started!')
    
    global timestamp_set, timecnt, tokenized_dataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f'------Epoch {epoch} started!')
        
        for i, batch in tqdm(enumerate(dataset), total = len(dataset)):
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
           
            # Ensure masks are of correct type
            src_embedding_mask = batch['attention_mask'].to(device).type(torch.bool)
            tgt_embedding_mask = batch['tgt_embedding_mask'].to(device).type(torch.bool)
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device).type(torch.bool)  # tgt.size(1) 
            # print(f'src: {src[0]} \n src_size: {src.size()}')
            # print(f'tgt: {tgt} \n tgt_size: {tgt.size()}')
            # print(f'src_embedding_mask: {~src_embedding_mask[0]} \n src_embedding_mask_size: {src_embedding_mask.size()}')
            # print(f'tgt_embedding_mask: {~tgt_embedding_mask} \n tgt_embedding_mask_size: {tgt_embedding_mask.size()}')
            # print(f'tgt_mask: {tgt_mask} \n tgt_mask_size: {tgt_mask.size()}')
            optimizer.zero_grad()
            output = model(src, tgt, ~src_embedding_mask, tgt_mask, ~tgt_embedding_mask)
            # print(output)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()  #batch loss 
            # print(loss.item())

            if i % 100 == 0 and i != 0:
                print(f'------Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
        avg_loss = total_loss  / len(dataset)  
        print(f'Epoch {epoch} complete. Average Loss: {avg_loss} Training TIme: {getNowTimeInterval():.2f} seconds')
        
    return model

vocab_size = tokenizer.vocab_size
d_model = 256
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 1024
max_seq_length = 256

print("Initializing Model and DataLoader...")
transformer_model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)
train_dataloader = DataLoader(CustomDataset(train_dataset), batch_size=32, shuffle=True)
print(f'------ Initizalization time: {getNowTimeInterval():.2f} seconds' )

trained_model = train(transformer_model, train_dataloader, epochs=2, batch_size=32, lr=0.0001)

def evaluate(model, dataloader, tokenizer):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            
            src_embedding_mask = batch['attention_mask'].to(device).type(torch.bool)
            tgt_embedding_mask = batch['tgt_embedding_mask'].to(device).type(torch.bool)
            tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device).type(torch.bool)  # tgt.size(1) 
            
            output = model(src, tgt, ~src_embedding_mask, tgt_mask, ~tgt_embedding_mask)
            output = output.argmax(dim=-1)
            
            for i in range(output.size(0)):
                ref = tokenizer.decode(tgt[i], skip_special_tokens=True)
                hyp = tokenizer.decode(output[i], skip_special_tokens=True)
                
                references.append([ref.split()])
                hypotheses.append(hyp.split())
                
    bleu_score = corpus_bleu(references, hypotheses)
    print(f'------BLEU score: {bleu_score* 100} ')
    return bleu_score

print("Evalution Started!")

valid_dataloader = DataLoader(CustomDataset(val_dataset), batch_size=32, shuffle=False)
evaluate(trained_model, valid_dataloader, tokenizer)



