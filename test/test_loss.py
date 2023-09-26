from unittest import TestCase

import torch

from vish.model.common.loss import bnf_embedding_cluster_loss, bnf_embedding_loss


class Test(TestCase):
    def setUp(self) -> None:
        self.broad_outputs = [torch.randn(32, 1, 768), torch.randn(32, 2)]
        self.fine_outputs = [torch.randn(32, 1, 768), [torch.randn(32, 10)]]
        self.broad_labels = torch.randint(0, 2, (32,))
        self.fine_labels = torch.randint(0, 10, (32,))
        self.fine_criterion = torch.nn.CrossEntropyLoss()
        self.p = 1.0

    def test_bnf_embedding_loss(self):
        print(
            "Broad Embedding Loss is",
            bnf_embedding_loss(
                self.broad_outputs,
                self.fine_outputs,
                self.broad_labels,
                self.fine_labels,
                self.fine_criterion,
                self.p,
            ),
        )

    def test_bnf_embedding_cluster_loss(self):
        print(
            "Broad Cluster Loss is",
            bnf_embedding_cluster_loss(
                self.broad_outputs,
                self.fine_outputs,
                self.broad_labels,
                self.fine_labels,
                self.fine_criterion,
                self.p,
            ),
        )