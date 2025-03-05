import pickle
import re
from abc import ABC, abstractmethod

import torch


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class GeneratorConnector(ABC):
    def __init__(self, generator, model, device):
        self.generator = generator
        self.model = model
        self.device = device

        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass


class StyleGeneratorConnector(GeneratorConnector):
    @abstractmethod
    def _load_model(self):
        pass

    def get_style_weights(self):
        weights = {k: v for k, v in self.generator.named_parameters()
                   if re.match("synthesis\.b[0-9]+\.[A-Za-z]+[0-1]+\.affine\.weight", k)}
        return weights  # []

    def get_fc_weights(self):
        weights = {k: v for k, v in self.generator.named_parameters() if 'mapping' in k and 'weight' in k}
        return weights

    def forward_z_w(self, z):
        x = z.to(torch.float32)
        x = normalize_2nd_moment(x)

        # Execute layers.
        for idx in range(self.generator.mapping.num_layers):
            x = self.generator.mapping.__getattr__(f'fc{idx}')(x)
        return x

    def forward_w_ws(self, w, update_emas):
        # Update moving average of W.
        if update_emas:
            self.generator.mapping.w_avg.copy_(w.detach().mean(dim=0).lerp(self.generator.w_avg,
                                                                           self.generator.w_avg_beta))

        # Broadcast and apply truncation.
        ws = w.unsqueeze(1).repeat([1, self.generator.mapping.num_ws, 1])
        return ws

    @abstractmethod
    def forward_w_img(self, w, truncation_psi, truncation_cutoff):
        pass

    @abstractmethod
    def forward_ws_img(self, ws, trunc_psi, trunc_cutoff, update_emas=False, **synthesis_kwargs):
        pass


class StyleGAN2Connector(StyleGeneratorConnector, ABC):
    """

    """
    def _load_model(self):
        with open(self.model, 'rb') as f:
            self.generator = pickle.load(f)['G_ema']
        self.generator.to(self.device)

    def forward_to_beyond(self, ws, trunc_psi, trunc_cutoff, update_emas=False, **synthesis_kwargs):
        """
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

        Args:
            ws:
            trunc_psi:
            trunc_cutoff:
            update_emas:
            **synthesis_kwargs:

        Returns:

        """
        if trunc_psi != 1:
            ws[:, :trunc_cutoff] = self.generator.mapping.w_avg.lerp(ws[:, :trunc_cutoff], trunc_psi)

        block_ws = []
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in self.generator.synthesis.block_resolutions:
            block = getattr(self.generator.synthesis, f'b{res}')
            block_code = ws.narrow(1, w_idx, block.num_conv + block.num_torgb)
            block_ws.append(block_code)
            w_idx += block.num_conv
            # print(res, w_idx, block_code.shape)

        x = img = None
        for res, cur_ws in zip(self.generator.synthesis.block_resolutions, block_ws):
            block = getattr(self.generator.synthesis, f'b{res}')
            print(block)
            x, img = block(x, img, cur_ws, **synthesis_kwargs)
            print(x.shape, img.shape)

        return self.generator.synthesis(ws, update_emas=update_emas, noise_mode='none', **synthesis_kwargs)

    def forward_w_img(self, w, trunc_psi, trunc_cutoff, update_emas=False, layers=None, **synthesis_kwargs):
        # Update moving average of W.
        if update_emas:
            self.generator.mapping.w_avg.copy_(w.detach().mean(dim=0).lerp(self.generator.w_avg,
                                                                           self.generator.w_avg_beta))

        # Broadcast and apply truncation.
        ws = w.unsqueeze(1).repeat([1, self.generator.mapping.num_ws, 1])
        if trunc_psi != 1:
            ws[:, :trunc_cutoff] = self.generator.mapping.w_avg.lerp(ws[:, :trunc_cutoff], trunc_psi)

        return self.generator.synthesis(ws, update_emas=update_emas, noise_mode='none', **synthesis_kwargs)

    def forward_ws_img(self, ws, trunc_psi, trunc_cutoff, update_emas=False, **synthesis_kwargs):
        if trunc_psi != 1:
            ws[:, :trunc_cutoff] = self.generator.mapping.w_avg.lerp(ws[:, :trunc_cutoff], trunc_psi)

        return self.generator.synthesis(ws, update_emas=update_emas, noise_mode='none', **synthesis_kwargs)

    def forward_wp_img(self, wp, trunc_psi, trunc_cutoff, update_emas):
        pass
