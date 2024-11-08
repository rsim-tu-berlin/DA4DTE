# import packages
from typing import Optional

from flask import Flask, request, jsonify, render_template

import numpy as np
import tifffile

import pytorch_lightning as pl
import torch
import typer

from configilm.ConfigILM import _get_hf_model as get_huggingface_model
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.util import huggingface_tokenize_and_pad

from lit4rsvqa import LitVisionEncoder


def load_tif_images_as_tensor(file_path):
    tif_stack = tifffile.imread(file_path)
    tif_stack_float32 = tif_stack.astype(np.float32)
    torch_tensor = torch.from_numpy(tif_stack_float32)
    return torch_tensor

def process_image(image_tensor):
    image_tensor = np.transpose(image_tensor, (2, 0, 1))
    image_tensor = image_tensor[:, :120, :120]

    pad_width = ((0, 10 - image_tensor.shape[0]), (0, 0), (0, 0))
    image_tensor = np.pad(image_tensor, pad_width, mode='constant', constant_values=0)
    image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
    image_tensor = torch.tensor(image_tensor.astype(np.float32))

    return image_tensor


def main(
    server_port: int,
    model_checkpoint_path: str,
    vision_model: str = "mobilevit_s",
    text_model: str = "prajjwal1/bert-tiny",
    seed: int = 42,
    matmul_precision: str = "medium",
):
    torch.set_float32_matmul_precision(matmul_precision)

    pl.seed_everything(seed, workers=True)

    model = LitVisionEncoder.load_from_checkpoint(model_checkpoint_path)
    model = model.to("cuda")
    print(
        f"Model Stats: Params: {model.get_stats()['params']:15,d}\n"
        f"              Flops: {model.get_stats()['flops']:15,d}"
    )

    hf_tokenizer, _ = get_huggingface_model(
        model_name=text_model, load_pretrained_if_available=False
    )

    print("=== Loading finished ===")
    
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return render_template('index.html')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Get image file and string from the request
            file = request.files['image']
            question = request.form['string']

            # Save the file temporarily
            file_path = '/tmp/uploaded_image.tif'
            file.save(file_path)

            image_tensor = load_tif_images_as_tensor(file_path)
            image_tensor = image_tensor.permute(2, 0, 1)  
            # image_tensor = process_image(image_tensor)
            image_tensor = image_tensor.to("cuda")
            tokenizer =  torch.tensor(huggingface_tokenize_and_pad(hf_tokenizer, question, 32))
            tokenizer = tokenizer.to("cuda")
            # Make prediction
            model.eval()
            with torch.no_grad():
                output = model((torch.unsqueeze(image_tensor, dim=0), torch.unsqueeze(tokenizer, dim=0)))

            prediction = "yes" if torch.max(output, dim=1)[0] > 0.5 else "no"

            return jsonify({'prediction': prediction})

        except Exception as e:
            return jsonify({'error': str(e)})

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=server_port)


if __name__ == "__main__":
    typer.run(main)
