from typing import Optional, Any, Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC, AveragePrecision


def init_metrics_fn(task_type: str, device: Optional[torch.device] = None, num_outputs: Optional[int] = None):
    if device is None:
        device = torch.device('cpu')

    if task_type == 'regression':
        return RegressionMetrics(device, num_outputs)
    elif task_type == 'binary_classification':
        return BinaryClassificationMetrics(device)
    elif task_type == 'multiclass_classification':
        return MulticlassClassificationMetrics(device, num_outputs)
    elif task_type == 'ordinal_classification':
        return OrdinalClassificationMetrics(device, num_outputs)
    raise ValueError(f"Unsupported task_type: {task_type!r}")


def _nan() -> Tensor:
    # Metric values are always tensors: callers do `value.mean().item()` when logging,
    # which a bare Python float does not support.
    return torch.tensor(float('nan'))


class _PresentClassMixin:
    """Macro-average only over the classes that actually appear in the targets.

    Metric objects are always built for the *training* split's class count N, because
    the task head's output index k means the train label space and must not be remapped
    per split. When a val/test split contains only M < N of those classes, torchmetrics
    scores every absent class as 0.0 and still divides by N, which silently deflates
    every macro average (a 5-class metric over 3 present classes reports 0.57 where the
    honest figure is 0.96).

    Subclasses record which classes were seen in the targets and reduce the per-class
    vectors over that mask instead — the same guard ``OrdinalMAE.compute`` already
    applies. "Present" means *appearing in the targets*: a class absent from the targets
    but occasionally predicted is left out of the macro, which slightly flatters
    ``precision``, though its false positives still cost the true classes their recall.
    """

    # Metrics built with average=None, reported as a macro over present classes.
    _MACRO_METRICS = ('balanced_accuracy', 'f1', 'precision', 'recall', 'auroc', 'auprc')
    # Of those, the ones whose per-class vector is also worth logging.
    _PER_CLASS_REPORTED = ('auroc', 'auprc')
    # Reported so every macro average carries the divisor it was taken over.
    _PRESENCE_COUNT_KEY = 'n_classes_present'

    def _init_presence(self, num_entries: int) -> None:
        self._num_entries = num_entries
        self._reset_presence()

    def _reset_presence(self) -> None:
        self._present = torch.zeros(self._num_entries, dtype=torch.bool)

    def _mark_present(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        idx = indices.detach().flatten().cpu().long().unique()
        idx = idx[(idx >= 0) & (idx < self._num_entries)]
        self._present[idx] = True

    def _macro_over_present(self, per_class: Any) -> Tensor:
        if not torch.is_tensor(per_class) or per_class.dim() == 0:
            return per_class  # already a scalar (e.g. the no-data NaN)
        present = self._present.to(per_class.device)
        if not bool(present.any()):
            return _nan()
        return per_class[present].mean()

    def _reduce(self, name: str, value: Any) -> Any:
        if name in self._MACRO_METRICS:
            return self._macro_over_present(value)
        return value

    def _extra_metrics(self, raw: dict) -> dict:
        extra = {
            f'{name}_per_output': raw[name]
            for name in self._PER_CLASS_REPORTED
            if name in raw and torch.is_tensor(raw[name]) and raw[name].dim() == 1
        }
        extra[self._PRESENCE_COUNT_KEY] = torch.tensor(float(self._present.sum()))
        return extra


class Metrics():
    def __init__(
        self,
        device: Optional[torch.device] = torch.device('cpu')
    ):
        self.device = device
        self.metrics = {}
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}
    
    def update(self, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None, pred_is_probs: bool = False) -> None:
        """Update all metrics with one batch.

        Args:
            pred: model outputs. Logits by default; already-normalised
                probabilities when ``pred_is_probs`` is True (e.g. the
                ``predict_proba`` output of a scikit-learn KNN probe).
            label: ground-truth labels.
            mask: optional validity mask (see subclass implementations).
            pred_is_probs: skip the logits -> probabilities conversion.
        """
        self._track_presence(label, mask)
        for name, metric in self.metrics.items():
            pred_proc, label_proc = self._prepare_inputs(name, pred, label, mask, pred_is_probs)
            if pred_proc is None:
                continue
            metric.update(pred_proc, label_proc)
            self._metric_has_data[name] = True
        
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None, pred_is_probs: bool = False):
        # Default: do nothing
        return pred, label

    def _track_presence(self, label: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        # Default: nothing to track. Classification subclasses record which classes
        # actually appear in the targets so that macro averages can skip the rest.
        pass

    def _reduce(self, name: str, value: Any) -> Any:
        # Default: report what the metric computed. Classification subclasses reduce
        # per-class vectors to a macro average over the classes present in the targets.
        return value

    def _extra_metrics(self, raw: dict) -> dict:
        # Extra entries to report alongside self.metrics, derived from the raw
        # (un-reduced) computed values (e.g. per-class breakdowns).
        return {}

    def _safe_compute_metric(self, metric: Metric):
        # Disable torchmetrics' dist.all_gather — metrics here are CPU-only
        # (KNN probe) and NCCL does not support CPU collectives.
        # The compute() wrapper reads _to_sync, not sync_on_compute directly.
        metric._to_sync = False
        metric.sync_on_compute = False
        try:
            return metric.compute()
        except ValueError as e:
            error_msg = str(e).lower()
            # Some regression/correlation metrics require at least 2 samples.
            if 'at least two samples' in error_msg:
                return _nan()
            raise

    def compute(self) -> dict:
        # Raw (un-reduced) values first: subclasses derive per-class breakdowns from
        # the same accumulators, so nothing is computed twice.
        raw = {
            name: self._safe_compute_metric(metric) if self._metric_has_data[name] else _nan()
            for name, metric in self.metrics.items()
        }
        metrics_dict = {name: self._reduce(name, value) for name, value in raw.items()}
        metrics_dict.update(self._extra_metrics(raw))
        return metrics_dict

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self._metric_has_data = {name: False for name in self._metric_has_data}
        self._reset_presence()

    def _reset_presence(self) -> None:
        # Default: no presence state to clear.
        pass

    def get(self, name: str) -> float:
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not found. Available: {list(self.metrics.keys())}")
        # Route through _reduce so get() can never disagree with compute().
        return self._reduce(name, self._safe_compute_metric(self.metrics[name]))

class RegressionMetrics(Metrics):
    def __init__(
        self,
        device: Optional[torch.device] = torch.device('cpu'),
        num_outputs: Optional[int] = None
    ):
        super().__init__(device)

        if num_outputs is None:
            self.num_outputs = 1
        else:
            self.num_outputs = num_outputs

        self.metrics = {
            'mae': MeanAbsoluteError().to(device),
            'mse': MeanSquaredError().to(device),
            'r2': R2Score(multioutput='uniform_average').to(device), # num_outputs is not automatically inferred from the shape of input tensors
            'pcc': PearsonCorrCoef(num_outputs=self.num_outputs).to(device)
        }
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}

        if num_outputs > 1:
            self.metrics['r2_per_output'] = R2Score(multioutput='raw_values').to(device)
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None, pred_is_probs: bool = False):
        """
        mask: 1 for NaN (ignore), 0 for valid (calculate).
        """
        if mask is None:
            if not (label == -100).any():
                return pred, label
            mask = ~(label == -100) # (B, num_outputs). True where valid, False where invalid
        else:
            mask = ~(mask.bool()) # True where valid, False where invalid
        
        if mask.sum() < 2: # not enough valid samples to compute metric, e.g. r2 requires at least 2 samples
            return None, None # handled by Metrics class
        
        pred_valid = pred[mask]
        label_valid = label[mask]

        # R² and PCC are undefined when all labels are identical
        if name in ('r2', 'r2_per_output', 'pcc'):
            if label_valid.unique().numel() == 1:
                return None, None  # will produce nan in compute()

        return pred_valid, label_valid # (num_valid,)

class BinaryClassificationMetrics(_PresentClassMixin, Metrics):
    # Binary metrics are already single-valued; only balanced_accuracy is a macro.
    _MACRO_METRICS = ('balanced_accuracy',)
    _PER_CLASS_REPORTED = ()

    def __init__(
        self,
        device = torch.device('cpu')
    ):
        super().__init__(device)
        self._init_presence(2)

        self.metrics = {
            'accuracy': Accuracy(task='binary').to(device),
            'balanced_accuracy': Accuracy(task='multiclass', average=None, num_classes=2).to(device),
            'f1': F1Score(task='binary').to(device),
            'precision': Precision(task='binary').to(device),
            'recall': Recall(task='binary').to(device),
            'auroc': AUROC(task='binary').to(device),
            'auprc': AveragePrecision(task='binary').to(device),
        }
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}

    def _track_presence(self, label: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        self._mark_present(label.long())

    def _reduce(self, name: str, value: Any) -> Any:
        # With only one class in the targets there is nothing to rank against and no
        # positives to recover, so these are undefined rather than zero.
        if name in ('f1', 'precision', 'recall', 'auroc', 'auprc') and not bool(self._present.all()):
            return _nan()
        return super()._reduce(name, value)

    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None, pred_is_probs: bool = False): # pred: logits, or probabilities if pred_is_probs
        probs = pred if pred_is_probs else torch.sigmoid(pred)
        label = label.long()

        if name in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']:
            pred = (probs > 0.5).long()
        elif name in ['auroc', 'auprc']:
            pred = probs

        return pred, label

class MulticlassClassificationMetrics(_PresentClassMixin, Metrics):
    def __init__(
        self,
        device = torch.device('cpu'),
        num_classes: int = None
    ):
        super().__init__(device)

        if num_classes is None:
            self.num_classes = 1
        else:
            self.num_classes = num_classes

        self._init_presence(self.num_classes)

        # The class-averaged metrics use average=None and are reduced in _reduce over
        # the classes present in the targets; see _PresentClassMixin. accuracy is micro
        # and so is unaffected by absent classes.
        self.metrics = {
            'accuracy': Accuracy(task='multiclass', average='micro', num_classes=self.num_classes).to(device),
            'balanced_accuracy': Accuracy(task='multiclass', average=None, num_classes=self.num_classes).to(device),
            'f1': F1Score(task='multiclass', average=None, num_classes=self.num_classes).to(device),
            'precision': Precision(task='multiclass', average=None, num_classes=self.num_classes).to(device),
            'recall': Recall(task='multiclass', average=None, num_classes=self.num_classes).to(device),
            'auroc': AUROC(task='multiclass', average=None, num_classes=self.num_classes).to(device),
            'auprc': AveragePrecision(task='multiclass', average=None, num_classes=self.num_classes).to(device),
        }
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}

    def _track_presence(self, label: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        self._mark_present(label.argmax(dim=-1))

    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None, pred_is_probs: bool = False): # pred: logits, or probabilities if pred_is_probs
        # logits -> probabilities
        probs = pred if pred_is_probs else torch.softmax(pred, dim=-1)
        pred = probs
        label = label.long()
        label = label.argmax(dim=-1)

        return pred, label

class OrdinalClassificationMetrics(MulticlassClassificationMetrics):
    """Multiclass metrics plus OrdinalMAE, for ordered categorical targets.

    All metrics here, OrdinalMAE included, take per-class probabilities: OrdinalMAE
    decodes them to the expected class. OrdinalMAE needs no presence handling — its
    own compute() already masks classes with no samples.
    """

    def __init__(
        self,
        device = torch.device('cpu'),
        num_classes: int = None
    ):
        super().__init__(device, num_classes)

        self.metrics['ordinal_mae'] = OrdinalMAE(num_classes=self.num_classes, average='macro').to(device)
        self._metric_has_data['ordinal_mae'] = False


class OrdinalMAE(Metric):
    """Ordinal Mean Absolute Error for ordered categorical labels.

    Computes the mean absolute error between predicted and target ordinal classes,
    where classes are mapped to integer ranks (e.g. Normal=0, Mild=1, Moderate=2,
    Severe=3). Penalizes predictions proportionally to how far they are from the
    true severity grade, unlike standard accuracy which treats all errors equally.

    ``preds`` may be given either as hard class indices, shape ``(N,)``, or as
    per-class probabilities, shape ``(N, num_classes)``. Probabilities are
    decoded to the *expected* class ``sum_c c * p_c`` and the error is measured
    against that continuous value. Soft decoding uses the ordinal structure of
    the probability vector rather than only its argmax, so a model that puts its
    remaining mass on an adjacent grade scores better than one that puts it on a
    distant grade. The resulting value is not the MAE of the model's discrete
    decision, and is not comparable to argmax-decoded MAE reported elsewhere.

    Supports three averaging modes:
        - ``'micro'``: standard MAE averaged over all samples equally.
        - ``'macro'``: MAE computed per class then averaged, giving equal weight
          to each severity grade regardless of prevalence. Recommended for
          imbalanced datasets.
        - ``'weighted'``: MAE computed per class then averaged, weighted by the
          number of true samples per class.

    Args:
        num_classes:
            Total number of ordinal classes including Normal (e.g. 4 for
            Normal/Mild/Moderate/Severe).
        average:
            Averaging strategy. One of ``'micro'``, ``'macro'``, ``'weighted'``.
            Defaults to ``'macro'``.
        ignore_index:
            Optional class index to ignore (e.g. NaN class). If provided,
            samples with this target label are excluded from all computations.
        class_weights:
            Optional 1-D tensor of length ``num_classes`` to apply custom
            per-error-magnitude weights. When provided, the absolute error
            ``|pred - target|`` is multiplied by ``class_weights[target]``
            before averaging. Useful for penalising errors on severe cases more
            heavily. Defaults to ``None`` (uniform weights).
        kwargs:
            Additional keyword arguments passed to the base ``Metric`` class
            (e.g. ``dist_sync_on_step``).

    Example:
        >>> import torch
        >>> from echotab.util.metrics import OrdinalMAE
        >>> # Normal=0, Mild=1, Moderate=2, Severe=3
        >>> preds  = torch.tensor([0, 0, 1, 3])
        >>> target = torch.tensor([0, 2, 1, 3])
        >>> metric = OrdinalMAE(num_classes=4, average="macro")
        >>> metric.update(preds, target)
        >>> metric.compute()
        tensor(0.5000)

    Example with per-class probabilities (soft decoding):
        >>> probs = torch.tensor([[0.7, 0.3, 0.0, 0.0],
        ...                       [0.0, 0.6, 0.4, 0.0]])
        >>> target = torch.tensor([0, 1])
        >>> metric = OrdinalMAE(num_classes=4, average="micro")
        >>> metric.update(probs, target)   # expected classes: 0.3, 1.4
        >>> metric.compute()
        tensor(0.3500)

    Example with ignore_index (NaN class mapped to index 4):
        >>> metric = OrdinalMAE(num_classes=4, average="macro", ignore_index=4)
        >>> preds  = torch.tensor([0, 1, 2, 4])   # last sample is NaN
        >>> target = torch.tensor([0, 2, 2, 4])
        >>> metric.update(preds, target)
        >>> metric.compute()
        tensor(0.2500)
    """

    # Internal states declared as class-level hints; registered in __init__.
    sum_abs_error: Tensor          # per-class cumulative |pred - target|
    total: Tensor                  # per-class sample counts

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        average: Literal["micro", "macro", "weighted"] = "macro",
        ignore_index: Optional[int] = None,
        class_weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_classes < 2:
            raise ValueError("`num_classes` must be at least 2.")
        if average not in ("micro", "macro", "weighted"):
            raise ValueError(
                f"`average` must be one of 'micro', 'macro', 'weighted', "
                f"got '{average}'."
            )
        if class_weights is not None:
            if class_weights.shape != (num_classes,):
                raise ValueError(
                    f"`class_weights` must have shape ({num_classes},), "
                    f"got {tuple(class_weights.shape)}."
                )

        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index

        # Register class_weights as a buffer so it moves with .to(device)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights: Optional[Tensor] = None

        # Distributed-safe accumulators: one slot per class
        self.add_state(
            "sum_abs_error",
            default=torch.zeros(num_classes, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _in_range(self, name: str, t: Tensor) -> None:
        valid_mask = t != self.ignore_index if self.ignore_index is not None else torch.ones_like(t, dtype=torch.bool)
        out_of_range = ((t[valid_mask] < 0) | (t[valid_mask] >= self.num_classes)).any()
        if out_of_range:
            raise ValueError(
                f"`{name}` contains values outside [0, {self.num_classes - 1}]."
            )

    def _validate_inputs(self, preds: Tensor, target: Tensor) -> None:
        if target.ndim != 1:
            raise ValueError(
                f"`target` must be a 1-D tensor of class indices, got "
                f"ndim={target.ndim}."
            )
        self._in_range("target", target)

        if preds.ndim == 2:
            # per-class probabilities
            if preds.shape != (target.shape[0], self.num_classes):
                raise ValueError(
                    f"2-D `preds` are read as per-class probabilities and must "
                    f"have shape ({target.shape[0]}, {self.num_classes}), got "
                    f"{tuple(preds.shape)}."
                )
        elif preds.ndim == 1:
            # hard class indices
            if preds.shape != target.shape:
                raise ValueError(
                    f"`preds` and `target` must have the same shape, "
                    f"got {tuple(preds.shape)} and {tuple(target.shape)}."
                )
            self._in_range("preds", preds)
        else:
            raise ValueError(
                f"`preds` must be a 1-D tensor of class indices or a 2-D tensor "
                f"of per-class probabilities, got ndim={preds.ndim}."
            )

    # ------------------------------------------------------------------
    # Core torchmetrics interface
    # ------------------------------------------------------------------

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate predictions and targets.

        Args:
            preds:  either a 1-D integer tensor of predicted class indices,
                    shape (N,), or a 2-D float tensor of per-class
                    probabilities, shape (N, num_classes), which is decoded to
                    the expected class ``sum_c c * p_c``.
            target: 1-D integer tensor of true class indices,      shape (N,).
        """
        self._validate_inputs(preds, target)

        # Mask out ignored indices (row-wise, so it also covers 2-D probabilities)
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            preds  = preds[mask]
            target = target[mask]

        if target.numel() == 0:
            return

        target = target.long()

        if preds.ndim == 2:
            # probabilities -> expected class (soft decoding)
            ranks = torch.arange(self.num_classes, device=preds.device, dtype=torch.float)
            preds = (preds.float() * ranks).sum(dim=-1)
        else:
            preds = preds.float()

        abs_error = (preds - target).abs().float()

        # Optionally weight errors by the true class
        if self.class_weights is not None:
            abs_error = abs_error * self.class_weights[target]

        # Accumulate per true class
        for c in range(self.num_classes):
            class_mask = target == c
            self.sum_abs_error[c] += abs_error[class_mask].sum()
            self.total[c]         += class_mask.sum()

    def compute(self) -> Tensor:
        """Compute and return the final OrdinalMAE scalar.

        Returns:
            Scalar tensor with the aggregated ordinal MAE.

        Raises:
            RuntimeError: if no samples have been accumulated yet.
        """
        if self.total.sum() == 0:
            raise RuntimeError(
                "OrdinalMAE.compute() called before any updates. "
                "Call update() with at least one batch first."
            )

        if self.average == "micro":
            return self.sum_abs_error.sum() / self.total.sum().float()

        # Per-class MAE; guard against empty classes with NaN then ignore them
        per_class_mae = torch.where(
            self.total > 0,
            self.sum_abs_error / self.total.float(),
            torch.tensor(float("nan"), device=self.sum_abs_error.device),
        )
        present = self.total > 0  # mask for classes that actually appeared

        if self.average == "macro":
            return per_class_mae[present].mean()

        # weighted: weight by number of true samples per class
        weights = self.total[present].float()
        return (per_class_mae[present] * weights).sum() / weights.sum()

    # ------------------------------------------------------------------
    # Optional convenience
    # ------------------------------------------------------------------

    def per_class_compute(self) -> Tensor:
        """Return per-class MAE tensor of shape (num_classes,).

        Classes with no samples are reported as ``float('nan')``.
        Useful for per-phenotype breakdowns in a results table.
        """
        return torch.where(
            self.total > 0,
            self.sum_abs_error / self.total.float(),
            torch.tensor(float("nan"), device=self.sum_abs_error.device),
        )