from pair_prediction.evaluation.base import base_eval
from pair_prediction.evaluation.sincfold import sincfold_eval
from pair_prediction.evaluation.spot_rna import spotrna_eval
from pair_prediction.evaluation.ufold import ufold_eval
from pair_prediction.evaluation.utils import collect_and_save_metrics

EVAL_FUNCTIONS = {
    'rinalmo': base_eval,
    'sincfold': sincfold_eval,
    'spotrna': spotrna_eval,
    'ufold': ufold_eval,
}
