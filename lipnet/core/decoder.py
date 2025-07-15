import numpy as np
import tensorflow.keras.backend as K

def _decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    decoded = K.ctc_decode(y_pred, input_length,
                           greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    return decoded[0][0].numpy()  # Get the best path (top 1)

def decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1, **kwargs):
    language_model = kwargs.get('language_model', None)

    paths = _decode(y_pred, input_length, greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    if language_model is not None:
        raise NotImplementedError("Language model search is not implemented yet")
    else:
        return paths

class Decoder:
    def __init__(self, greedy=True, beam_width=100, top_paths=1, **kwargs):
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = top_paths
        self.language_model = kwargs.get('language_model', None)
        self.postprocessors = kwargs.get('postprocessors', [])

    def decode(self, y_pred, input_length):
        decoded = decode(y_pred, input_length, greedy=self.greedy,
                         beam_width=self.beam_width, top_paths=self.top_paths,
                         language_model=self.language_model)
        return decoded
