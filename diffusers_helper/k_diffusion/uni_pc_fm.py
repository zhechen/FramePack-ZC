# Better Flow Matching UniPC by Lvmin Zhang
# (c) 2025
# CC BY-SA 4.0
# Attribution-ShareAlike 4.0 International Licence


import sys

import torch

from typing import Any, Dict, Optional

from tqdm import trange

from framepack import flf_helpers


def maybe_prepare_endpoints(pipeline, cfg: Dict[str, Any], device, scale: float, size_hw=None):
    start_path = cfg.get('start_frame')
    end_path = cfg.get('end_frame')
    k = int(cfg.get('endpoint_frames', 0))
    if not start_path and not end_path:
        return None

    schedule = flf_helpers.make_schedule(k, cfg.get('endpoint_schedule', 'cosine')) if k > 0 else torch.zeros(0)
    ctx: Dict[str, Any] = {
        'endpoint_frames': k,
        'endpoint_strength': float(cfg.get('endpoint_strength', 0.65)),
        'endpoint_mode': cfg.get('endpoint_mode', 'blend'),
        'schedule': schedule,
        'frame_idx': int(cfg.get('frame_idx', 0)),
        'total_frames': int(cfg.get('total_frames', 0)),
        'two_pass_meet': bool(cfg.get('two_pass_meet', False)),
    }

    def _encode(path):
        image = flf_helpers.load_image(path, size_hw=size_hw)
        latent = flf_helpers.encode_vae(pipeline.vae, image, device=device, scale=scale)
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        return latent

    ctx['z_start'] = _encode(start_path) if start_path else None
    ctx['z_end'] = _encode(end_path) if end_path else None

    return ctx


def apply_endpoint_controls(
    frame_idx: int,
    total_frames: int,
    x_t: torch.Tensor,
    x0_pred: torch.Tensor,
    sigma: torch.Tensor,
    z_start: Optional[torch.Tensor],
    z_end: Optional[torch.Tensor],
    s_end_sched: Optional[torch.Tensor],
    cfg: Dict[str, Any],
    pipeline,
):
    frames = int(cfg.get('endpoint_frames', 0))
    if frames <= 0:
        return x_t, x0_pred

    if s_end_sched is None or s_end_sched.numel() == 0:
        return x_t, x0_pred

    strength = float(cfg.get('endpoint_strength', 0.65))
    mode = cfg.get('endpoint_mode', 'blend')

    span = x0_pred.shape[2] if x0_pred.ndim >= 5 else 1
    device = x0_pred.device
    indices = torch.arange(span, device=device) + frame_idx
    schedule = s_end_sched.to(device=device, dtype=x0_pred.dtype)

    def _expand(latent):
        if latent is None:
            return None
        target = latent.to(device=device, dtype=x0_pred.dtype)
        if target.ndim == 4:
            target = target.unsqueeze(2)
        if target.shape[2] == 1 and span > 1:
            target = target.expand(-1, -1, span, -1, -1)
        return target

    z_start = _expand(z_start)
    z_end = _expand(z_end)

    sigma_view = sigma.to(device=x_t.device, dtype=x_t.dtype)
    while sigma_view.ndim < x_t.ndim:
        sigma_view = sigma_view.view(*sigma_view.shape, 1)
    sigma_view = sigma_view.clamp(min=1e-5)

    if z_start is not None:
        start_mask = indices < frames
        if start_mask.any():
            gather_idx = indices[start_mask]
            alpha = torch.zeros(span, device=device, dtype=x0_pred.dtype)
            alpha[start_mask] = schedule.index_select(0, gather_idx.clamp_max(schedule.numel() - 1)) * strength
            alpha_view = alpha.view(1, 1, span, 1, 1)
            if mode == 'lock':
                noise = (x_t - x0_pred) / sigma_view
                z_locked = z_start
                x0_pred = torch.where(alpha_view > 0, z_locked, x0_pred)
                z_noised = flf_helpers.noise_latent_to_sigma(pipeline, z_locked, sigma, noise=noise)
                x_t = torch.where(alpha_view > 0, z_noised, x_t)
            else:
                x0_pred = flf_helpers.anchor_x0(x0_pred, z_start, alpha_view)

    if z_end is not None and total_frames > 0:
        tail = total_frames - 1 - indices
        end_mask = tail < frames
        if end_mask.any():
            gather_idx = tail[end_mask].clamp_max(schedule.numel() - 1)
            alpha = torch.zeros(span, device=device, dtype=x0_pred.dtype)
            alpha[end_mask] = schedule.index_select(0, gather_idx) * strength
            alpha_view = alpha.view(1, 1, span, 1, 1)
            x0_pred = flf_helpers.anchor_x0(x0_pred, z_end, alpha_view)

    return x_t, x0_pred


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


class FlowMatchUniPC:
    def __init__(self, model, extra_args, variant='bh1', pipeline=None, endpoint_context: Optional[Dict[str, Any]] = None):
        self.model = model
        self.variant = variant
        self.extra_args = extra_args
        self.pipeline = pipeline
        self.endpoint_context = endpoint_context

    def model_fn(self, x, t):
        return self.model(x, t, **self.extra_args)

    def update_fn(self, x, model_prev_list, t_prev_list, t, order):
        assert order <= len(model_prev_list)
        dims = x.dim()

        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = - torch.log(t_prev_0)
        lambda_t = - torch.log(t)
        model_prev_0 = model_prev_list[-1]

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = - torch.log(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h[0]
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError('Bad variant!')

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        use_predictor = len(D1s) > 0

        if use_predictor:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], device=b.device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None
            rhos_p = None

        if order == 1:
            rhos_c = torch.tensor([0.5], device=b.device)
        else:
            rhos_c = torch.linalg.solve(R, b)

        x_t_ = expand_dims(t / t_prev_0, dims) * x - expand_dims(h_phi_1, dims) * model_prev_0

        if use_predictor:
            pred_res = torch.tensordot(D1s, rhos_p, dims=([1], [0]))
        else:
            pred_res = 0

        x_t = x_t_ - expand_dims(B_h, dims) * pred_res
        model_t = self.model_fn(x_t, t)

        if D1s is not None:
            corr_res = torch.tensordot(D1s, rhos_c[:-1], dims=([1], [0]))
        else:
            corr_res = 0

        D1_t = (model_t - model_prev_0)
        x_t = x_t_ - expand_dims(B_h, dims) * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    def sample(self, x, sigmas, callback=None, disable_pbar=False):
        order = min(3, len(sigmas) - 2)
        model_prev_list, t_prev_list = [], []
        for i in trange(
            len(sigmas) - 1,
            disable=disable_pbar,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            vec_t = sigmas[i].expand(x.shape[0])

            if i == 0:
                model_x = self.model_fn(x, vec_t)
                if self.endpoint_context is not None:
                    ctx = self.endpoint_context
                    x, model_x = apply_endpoint_controls(
                        ctx.get('frame_idx', 0),
                        ctx.get('total_frames', 0),
                        x,
                        model_x,
                        vec_t,
                        ctx.get('z_start'),
                        ctx.get('z_end'),
                        ctx.get('schedule'),
                        ctx,
                        self.pipeline,
                    )
                model_prev_list = [model_x]
                t_prev_list = [vec_t]
            elif i < order:
                init_order = i
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, init_order)
                if self.endpoint_context is not None:
                    ctx = self.endpoint_context
                    x, model_x = apply_endpoint_controls(
                        ctx.get('frame_idx', 0),
                        ctx.get('total_frames', 0),
                        x,
                        model_x,
                        vec_t,
                        ctx.get('z_start'),
                        ctx.get('z_end'),
                        ctx.get('schedule'),
                        ctx,
                        self.pipeline,
                    )
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)
            else:
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, order)
                if self.endpoint_context is not None:
                    ctx = self.endpoint_context
                    x, model_x = apply_endpoint_controls(
                        ctx.get('frame_idx', 0),
                        ctx.get('total_frames', 0),
                        x,
                        model_x,
                        vec_t,
                        ctx.get('z_start'),
                        ctx.get('z_end'),
                        ctx.get('schedule'),
                        ctx,
                        self.pipeline,
                    )
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)

            model_prev_list = model_prev_list[-order:]
            t_prev_list = t_prev_list[-order:]

            if callback is not None:
                callback({'x': x, 'i': i, 'denoised': model_prev_list[-1]})

        return model_prev_list[-1]


def sample_unipc(
    model,
    noise,
    sigmas,
    extra_args=None,
    callback=None,
    disable=False,
    variant='bh1',
    pipeline=None,
    endpoint_context: Optional[Dict[str, Any]] = None,
):
    assert variant in ['bh1', 'bh2']
    return FlowMatchUniPC(
        model,
        extra_args=extra_args,
        variant=variant,
        pipeline=pipeline,
        endpoint_context=endpoint_context,
    ).sample(noise, sigmas=sigmas, callback=callback, disable_pbar=disable)
