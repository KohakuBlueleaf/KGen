from functools import partial
from time import time_ns
from transformers import PreTrainedModel


def generate(self: PreTrainedModel, *args, **kwargs):
    self.t0 = time_ns()
    self.t_forward = []
    result = self.org_generate(*args, **kwargs)
    self.t1 = time_ns()
    return result


def forward(self: PreTrainedModel, *args, **kwargs):
    t0 = time_ns()
    result = self.org_forward(*args, **kwargs)
    t1 = time_ns()
    self.t_forward.append((t0, t1))
    return result


def export_time(self: PreTrainedModel):
    prompt_process = self.t_forward[0][1] - self.t_forward[0][0]

    total_eval = 0
    total_sampling = 0
    prev_t1 = self.t_forward[0][1]
    for t0, t1 in self.t_forward[1:]:
        total_eval += t1 - t0
        total_sampling += t0 - prev_t1
        prev_t1 = t1
    total_eval += self.t1 - self.t_forward[-1][1]

    timings = {
        "total": (self.t1 - self.t0) / 1e6,
        "prompt_process": prompt_process / 1e6,
        "total_eval": total_eval / 1e6,
        "total_sampling": total_sampling / 1e6,
    }

    return timings


def patch(model: PreTrainedModel):
    model.org_forward = model.forward
    model.org_generate = model.generate
    model.forward = partial(forward, model)
    model.generate = partial(generate, model)
    model.export_time = partial(export_time, model)
    return model
