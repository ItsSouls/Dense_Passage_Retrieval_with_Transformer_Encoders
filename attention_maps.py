import hydra
import random
import torch
import matplotlib.pyplot as plt
from src.data_utils.corpus import Corpus
from src.retrievers.dpr import DPR

@hydra.main(config_path='config', config_name='train_experiment.yaml', version_base=None)
def plot_attention(cfg):
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    
    checkpoint_file = torch.load("outputs/train/C_lr3e-4/best_encoders.pt", map_location="cpu")
    
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')

    dpr = DPR(cfg)
    tokenizer = dpr.tokenizer
    query_encoder = dpr.query_encoder
    query_encoder.load_state_dict(checkpoint_file["query_encoder"])
    query_encoder.eval()

    print("Weights loaded")

    query = corpus.test['question'][2]
    print(query)

    encoded = tokenizer(
                query,    
                truncation=True,       
                max_length= cfg.model.max_query_length,       
                return_tensors="pt",
            )
    input_ids = encoded["input_ids"]
    attention_mask =  encoded["attention_mask"]
   
    with torch.no_grad():
        _, attentions = query_encoder(input_ids, attention_mask, return_attention=True)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    print(tokens)
    print(len(attentions), attentions[0].shape)

    fig, axes = plt.subplots(6, 4, figsize=(16, 24))

    for layer in range(6):
        for head in range(4):
            map = attentions[layer][0,head].numpy()
            ax = axes[layer,head]
            ax.set_title(f"Layer {layer} - Head {head}", fontsize=9)
            ax.imshow(map, cmap="viridis")
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=6)
            ax.imshow(map, cmap="viridis", vmin=0, vmax=1)

    plt.tight_layout()
    plt.savefig("attention_trained_query3.png", dpi=150)

if __name__ == "__main__":
    plot_attention()