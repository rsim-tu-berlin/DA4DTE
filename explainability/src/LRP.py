import torch
import numpy as np

class LayerWiseRelevancePropagation:
    def __init__(self, model, eye, dev):
        self.model = model
        self.eye = eye
        self.model.eval()
        self.dev = dev

    def run_relprop(
        self,
        input,
        index=None,
        method="transformer_attribution",
        is_ablation=False,
        start_layer=0,
    ):
        output = self.model(input)

        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.dev) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(
            self.eye,
            torch.tensor(one_hot_vector).to(output.device),
            method=method,
            is_ablation=is_ablation,
            start_layer=start_layer,
            **kwargs
        )