# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from fairseq.data import (
    encoders,
)
from fairseq.tasks import LegacyFairseqTask, register_task

@register_task("sentiment_analysis")
class SentimentAnalysisTask(LegacyFairseqTask):
    """
    Task for Aspect-Based Sentiment Analysis with external knowledge injection (K-Former).
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data", metavar="DIR", help="path to data directory; we load <split>.jsonl"
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument("--num-classes", type=int, default=3)

    def _init_(self, args, vocab, knowledge_embeddings):
        super()._init_(args)
        self.vocab = vocab
        self.knowledge_embeddings = knowledge_embeddings
        self.bpe = encoders.build_bpe(args) 

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load vocab and knowledge embeddings)."""
        vocab = cls.load_dictionary(os.path.join(args.data, "dict.txt"))
        knowledge_embeddings = cls.load_knowledge_embeddings("/Users/sameeksha/Desktop/dataset/Sem15_embeddings/hermit_ontology.embeddings")
        print("| dictionary: {} types".format(len(vocab)))
        print("| knowledge embeddings loaded")
        return cls(args, vocab, knowledge_embeddings)

    @classmethod
    def load_knowledge_embeddings(cls, filename):
        """Load the knowledge embeddings from the filename."""
        return torch.load(filename)

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        model.register_classification_head(
            "sentence_classification_head",
            num_classes=3,)
        return model

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab
