'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: Customized inference pipeline.
'''
import argparse

import json
import torch
from os.path import join
from pytorch_utils import get_model
from utils import save_ds_in_txt, read_json
from pytorch_utils import customize_predictor
from dataloader import get_text_encoder_decoder, customize_dataloader_func


parser = argparse.ArgumentParser()
parser.add_argument("--config_fp", default='config.json', type=str, help="The filepath to the config.json.")
args = parser.parse_args()

config = read_json(args.config_fp)
InferConfig = config["InferConfig"]
    

def main():
    out_dir = InferConfig["out_dir"]
    saved_model_fp = join(out_dir, "model.pt")
    ModelConfig = read_json(join(out_dir, "ModelConfig.json"))
    TrainConfig = read_json(join(out_dir, "TrainConfig.json"))

    padding_idx = 0
    in_vocab = TrainConfig["in_vocab"]
    out_vocab = TrainConfig["out_vocab"]
        
    in_seq_encoder, _ = get_text_encoder_decoder(in_vocab)
    out_seq_encoder, out_seq_decoder = get_text_encoder_decoder(out_vocab)
    dataloader_func = customize_dataloader_func(in_seq_encoder, 
                                                out_seq_encoder, 
                                                padding_idx)
    
    model = get_model(ModelConfig, None)
    model.load_state_dict(torch.load(saved_model_fp))
    
    batch_size = 1
    predictor = customize_predictor(model, dataloader_func, 
                                    out_seq_decoder, batch_size=batch_size)
    
    infer_text_fp = InferConfig["infer_text_fp"]
    texts = [t.strip() for t in open(infer_text_fp, "r")]

    if InferConfig["x_sep"]:
        ori_texts = texts.copy()
        texts = [t.split(InferConfig["x_sep"]) for t in texts]
    
    predicted = predictor(texts)

    if InferConfig["x_sep"]:
        texts = ori_texts

    if InferConfig["y_sep"]:
        predicted = [f'{InferConfig["y_sep"]}'.join(p) for p in predicted]
    else:
        predicted = [f''.join(p) for p in predicted]

    inferred_text_fp = infer_text_fp + "_predicted.txt"
    save_ds_in_txt(list(zip(texts, predicted)), inferred_text_fp)
    
    
if __name__ == "__main__":
    main()
