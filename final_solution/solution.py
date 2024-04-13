import typing as tp
import pathlib
import numpy as np
from onnxruntime import InferenceSession
from tokenizers import Tokenizer
import json

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]

PATH_TO_CHECKPOINT = pathlib.Path("final_solution") / 'checkpoint'
PATH_TO_TOKENIZER = PATH_TO_CHECKPOINT / 'pretrain/tokenizer.json'
PATH_TO_WEIGHTS = PATH_TO_CHECKPOINT / 'export.onnx'
PATH_TO_MAP = pathlib.Path("final_solution") / 'issuer_map.json'
THRESHOLD = 0.5

with open(PATH_TO_MAP) as json_data:
    id_to_index = json.load(json_data)
    index_to_id = dict(zip(
        id_to_index.values(),
        id_to_index.keys(),
    ))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """
    ort = InferenceSession(
        PATH_TO_WEIGHTS,
        providers=['CUDAExecutionProvider']
    )
    tokenizer = Tokenizer.from_file(
        str(PATH_TO_TOKENIZER)
    )

    output_scores = []

    for message in messages:
        current_scores = []
        enc_text = tokenizer.encode(message)
        inputs = {
            'input_ids': np.asarray([enc_text.ids]),
            'attention_mask': np.asarray([enc_text.attention_mask]),
        }
        ment_scores, sent_scores = ort.run(
            ['mention', 'sentiment'],
            inputs
        )
        ment_scores = sigmoid(ment_scores).squeeze()
        sent_scores = np.argmax(sent_scores, axis=-1).squeeze()

        for index, (ment_score, sent_score) in enumerate(zip(ment_scores, sent_scores)):
            if ment_score > THRESHOLD:
                entity_id = index_to_id[index]
                entity_score = float(sent_score)
                current_scores.append((entity_id, entity_score))
        output_scores.append(current_scores)
    return output_scores
