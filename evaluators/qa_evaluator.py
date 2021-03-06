import torch.nn.functional as F

from evaluators.evaluator import Evaluator
from utils.relevancy_metrics import get_map_mrr


class QAEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, embedding, data_loader, batch_size, device):
        super(QAEvaluator, self).__init__(dataset_cls, model, embedding, data_loader, batch_size, device)

    def get_scores(self):
        self.model.eval()
        test_cross_entropy_loss = 0
        qids = []
        true_labels = []
        predictions = []

        for batch in self.data_loader:
            qids.extend(batch.id.data.cpu().numpy())
            # Select embedding
            sent1 = self.embedding(batch.sentence_1).transpose(1, 2)
            sent2 = self.embedding(batch.sentence_2).transpose(1, 2)

            output = self.model(sent1, sent2, batch.ext_feats)
            test_cross_entropy_loss += F.cross_entropy(output, batch.label, size_average=False).data[0]

            true_labels.extend(batch.label.data.cpu().numpy())
            predictions.extend(output.data.exp()[:, 1].cpu().numpy())

            del output

        qids = list(map(lambda n: int(round(n * 10, 0)) / 10, qids))

        mean_average_precision, mean_reciprocal_rank = get_map_mrr(qids, predictions, true_labels, self.data_loader.device)
        test_cross_entropy_loss /= len(batch.dataset.examples)

        return [test_cross_entropy_loss, mean_average_precision, mean_reciprocal_rank], ['cross entropy loss', 'map', 'mrr']

    def get_final_prediction_and_label(self, batch_predictions, batch_labels):
        predictions = batch_predictions.exp()[:, 1]

        return predictions, batch_labels
