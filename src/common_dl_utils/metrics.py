""" 
Some utilities for collecting and reporting metrics during training
"""

import abc
from enum import Enum
from typing import Union

__all__ = [
    'MetricFrequency',
    'Metric',
    'MetricCollector',
    'add_prefix_to_dictionary_keys',
    'update_history',
]

class MetricFrequency(Enum):
    EVERY_BATCH = 'every_batch'
    EVERY_EPOCH = 'every_epoch'
    EVERY_N_BATCHES = 'every_n_batches'
    EVERY_N_EPOCHS = 'every_n_epochs'

class Metric(abc.ABC):
    """
    Abstract base class for metrics
    """
    frequency: MetricFrequency
    required_kwargs: set
    
    @abc.abstractmethod
    def compute(self, **kwargs)->dict:
        return {repr(self): None}
    
    def __repr__(self):
        return type(self).__name__
    
    def __str__(self):
        return self.__repr__()

class MetricCollector:
    def __init__(self, *metrics:Metric, batch_frequency:int, epoch_frequency:int):
        """
        MetricCollector object for collecting and managing metrics during training
        :param metrics: the Metrics to be collected
        :param batch_frequency: metrics with metric.frequency == MetricFrequency.Every_N_BATCHES will be logged every batch_frequency batches
        :param epoch_frequency: metrics with metric.frequency == MetricFrequency.Every_N_EPOCHS will be logged every epoch_frequency epochs
        """
        self.metrics = metrics
        self.batch_frequency = batch_frequency
        self.epoch_frequency = epoch_frequency

        self.on_every_batch = [
            metric for metric in self.metrics
            if metric.frequency.value == MetricFrequency.EVERY_BATCH.value
            # the reason for using .value is that if this module is loaded multiple times,
            # the enums will be different objects, and the comparison will fail without .value
            # see the warning in https://docs.python.org/3/howto/enum.html#comparisons
        ]

        self.on_every_epoch = [
            metric for metric in self.metrics
            if metric.frequency.value == MetricFrequency.EVERY_EPOCH.value
        ]

        self.on_every_n_batches = [
            metric for metric in self.metrics
            if metric.frequency.value == MetricFrequency.EVERY_N_BATCHES.value
        ]

        self.on_every_n_epochs = [
            metric for metric in self.metrics
            if metric.frequency.value == MetricFrequency.EVERY_N_EPOCHS.value
        ]
        
        self._step = 0
        self._epoch = 1

        self.required_kwargs_on_batch_end = set.union(*[metric.required_kwargs for metric in self.on_every_batch + self.on_every_n_batches])
        self.required_kwargs_on_epoch_end = set.union(*[metric.required_kwargs for metric in self.on_every_epoch + self.on_every_n_epochs])

    def on_batch_end(self, **kwargs):
        """
        Compute and report metrics after finishing a batch
        Also, upon every self.batch_frequency batches, compute and report metrics with metric.frequency == MetricFrequency.EVERY_N_BATCHES

        :param kwargs: keyword arguments to be passed to the metrics
            these depend on the metrics
        :return: dictionary of metric names and values
        """
        self._step += 1
        results = {
            'batch_within_epoch': self._step,
            'epoch': kwargs.get('epoch', self._epoch),
        }

        for metric in self.on_every_batch:
            results.update(metric.compute(**kwargs))

        if self._step % self.batch_frequency == 0:
            for metric in self.on_every_n_batches:
                results.update(metric.compute(**kwargs))
        
        return results
    
    def on_epoch_end(self, **kwargs):
        """
        Compute and report metrics after finishing an epoch
        Also, upon every self.epoch_frequency epochs, compute and report metrics with metric.frequency == MetricFrequency.EVERY_N_EPOCHS 

        :param kwargs: keyword arguments to be passed to the metrics
            these depend on the metrics
        :return: dictionary of metric names and values

        Typical kwargs required by the metrics are:
        - validation_loader
        - model
        - device
        """
        results = {
            'epoch': kwargs.get('epoch', self._epoch),
        }
        for metric in self.on_every_epoch:
            results.update(metric.compute(**kwargs))
        
        if self._epoch % self.epoch_frequency == 0:
            for metric in self.on_every_n_epochs:
                results.update(metric.compute(**kwargs))
        
        self._epoch += 1
        self._step = 0
        return results

    def __repr__(self):
        return f'MetricCollector(on_every_batch={self.on_every_batch}, on_every_epoch={self.on_every_epoch}, on_every_n_batches={self.on_every_n_batches}, on_every_n_epochs={self.on_every_n_epochs}, batch_frequency={self.batch_frequency}, epoch_frequency={self.epoch_frequency}, required_kwargs_on_batch_end={self.required_kwargs_on_batch_end}, required_kwargs_on_epoch_end={self.required_kwargs_on_epoch_end})'


def add_prefix_to_dictionary_keys(dictionary:dict, prefix:str, excluded_keys:Union[tuple, set, list]=('epoch', 'batch_within_epoch'))->dict:
    return {
        (f'{prefix}{key}' if key not in excluded_keys else key): value
        for key, value in dictionary.items()
    }

def update_history(history, updates):
    for key in history:
        history[key].append(updates[key])
