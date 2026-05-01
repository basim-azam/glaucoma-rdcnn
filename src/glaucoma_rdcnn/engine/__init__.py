"""Training / evaluation / inference engines."""

from glaucoma_rdcnn.engine.evaluator import Evaluator
from glaucoma_rdcnn.engine.inference import infer_image
from glaucoma_rdcnn.engine.trainer import Trainer

__all__ = ["Trainer", "Evaluator", "infer_image"]
