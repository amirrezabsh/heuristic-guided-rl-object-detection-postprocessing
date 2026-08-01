from __future__ import annotations

import inspect
import textwrap


def patch_validator_to_skip_loss_shape_errors() -> None:
    """Allow validation metrics to continue if dense-target validation loss fails.

    Ultralytics computes validation loss during training validation. On MPS with
    dense Global Wheat labels, target assignment can raise shape/broadcast
    errors while mAP postprocessing would still be valid. The patch skips only
    those validation-loss batches; training loss and validation metrics remain
    unchanged.
    """

    from ultralytics.engine import validator as validator_module

    if getattr(validator_module.BaseValidator, "_rl_skip_val_loss_patch", False):
        return

    source = textwrap.dedent(inspect.getsource(validator_module.BaseValidator.__call__))
    old = """if self.training:
                self.loss += model.loss(batch, preds)[1]"""
    new = """if self.training:
                try:
                    self.loss += model.loss(batch, preds)[1]
                except RuntimeError as exc:
                    message = str(exc)
                    if (
                        "shape mismatch" not in message
                        and "must match" not in message
                        and "broadcast" not in message
                    ):
                        raise"""
    if old not in source:
        raise RuntimeError("Could not patch Ultralytics validator loss block.")
    namespace = validator_module.__dict__
    exec(source.replace(old, new), namespace)
    validator_module.BaseValidator.__call__ = namespace["__call__"]
    validator_module.BaseValidator._rl_skip_val_loss_patch = True
