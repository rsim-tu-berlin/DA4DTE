import torch
import numpy as np


class LRP:
    def __init__(self, model, eye):
        self.model = model
        self.eye = eye
        self.model.eval()

    def run_relprop(
        self,
        modality,
        input,
        index=None,
        method="transformer_attribution",
        is_ablation=False,
        start_layer=0,
    ):

        foward_func = getattr(self.model, f"forward_{modality}")
        output = foward_func(input)

        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        assert one_hot.is_cuda

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(
            eye=self.eye,
            cam=torch.tensor(one_hot_vector).to(output.device),
            method=method,
            is_ablation=is_ablation,
            start_layer=start_layer,
            **kwargs,
        )
