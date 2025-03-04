import numpy as np

import torch
import torch.nn as nn

from modeling_cxrbert import CXRBertModel
from transformers import AutoTokenizer, AutoModel
import os


class TextExtractor:
    def __init__(self, 
                 base_model_name='microsoft/BiomedVLP-CXR-BERT-specialized', 
                 resume_model='./models/test_run2',
                 max_seq_length = 2048,
                 hidden_size = 768,
                 save_seq_len = 192
                ):
        self.base_model_name = base_model_name
        self.resume_model = resume_model
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.save_seq_len = save_seq_len

        # init model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(self.device)

        self.model = CXRBertModel.from_pretrained(self.base_model_name)
        # extend embeddings
        old_embed = self.model.bert.embeddings.position_embeddings.weight.data
        tmp_dim = old_embed.shape[0]
        #print("tmp_dim:", tmp_dim)
        self.model.bert.embeddings.position_embeddings = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.model.bert.embeddings.position_embeddings.weight.data[:tmp_dim, :] = old_embed
        self.model.bert.embeddings.register_buffer("position_ids", torch.arange(self.max_seq_length).expand((1, -1)))
        self.model.config.max_position_embeddings = self.max_seq_length

        ckpt = torch.load(self.resume_model+"/pytorch_model.bin", map_location=self.device)

        msg = self.model.load_state_dict(ckpt, strict=False)
        print(msg)
        self.model.to(self.device)
        self.model.eval()


        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)



    def run(self, text, output_folder, file_name):
        if not os.path.exists(output_folder):
            print("Createing output folder: " + output_folder)
            os.makedirs(output_folder)

        file_path = os.path.join(output_folder, file_name)
        tokens_path = os.path.join(output_folder, file_name.replace('.npy', '_tokens.npy'))
        if os.path.exists(file_path) and os.path.exists(tokens_path):
            print("File already exists: " + file_path + " and " + tokens_path)

            return


        example_0 = self.tokenize_function(text)
        for item in example_0.keys():
            example_0[item] = example_0[item].to(self.device)

        with torch.no_grad():
            feature_0 = self.model(**example_0)

        feature_0_np = feature_0.hidden_states[-1][:, :self.save_seq_len, :].detach().cpu().numpy()
        print(feature_0_np.shape)



        np.save(file_path, feature_0_np)
        print("Saved to: " + file_path)

        #save tokens
        tokens_np = example_0['input_ids'][:, :self.save_seq_len].detach().cpu().numpy()
        np.save(tokens_path, tokens_np)
        print("Saved rokens to: " + tokens_path)

    def tokenize_function(self, example_text):
        # Remove empty lines

            return self.tokenizer(
                example_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
                return_tensors='pt'
            )
"""
text_extractor = TextExtractor()
text = 'No consolidation is identified. No pulmonary nodules are noted. Bone windowed images demonstrate no lytic or blastic lesions.,No evidence of pulmonary embolus.'
output_folder = "./results/test"
file_name='no_consolidation.npy'
text_extractor.run(text, output_folder, file_name)
"""