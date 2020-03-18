import torch

from transformer_bach.data_processor import DataProcessor
from transformer_bach.utils import to_numpy


class BachDataProcessor(DataProcessor):
    def __init__(self, embedding_size, num_events, num_tokens_per_channel):
        super(BachDataProcessor, self).__init__(embedding_size=embedding_size,
                                                num_events=num_events,
                                                num_tokens_per_channel=num_tokens_per_channel
                                                )

    def postprocess(self, x):
        tensor_score = torch.cat(x, dim=0)
        tensor_score = to_numpy(tensor_score)
        return tensor_score


def get_data_processor(dataloader_generator,
                       data_processor_type,
                       data_processor_kwargs):
    if data_processor_type == 'bach':
        # compute num_events num_tokens_per_channel
        dataset = dataloader_generator.dataset
        num_events = dataset.sequences_size * dataset.subdivision
        num_tokens_per_channel = [len(d) for d in dataset.index2note_dicts]
        data_processor = BachDataProcessor(embedding_size=data_processor_kwargs['embedding_size'],
                                           num_events=num_events,
                                           num_tokens_per_channel=num_tokens_per_channel)
        return data_processor
