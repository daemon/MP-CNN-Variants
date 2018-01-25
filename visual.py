import numpy as np
import torch
import torch.autograd as autograd

class SimilarityVisualizer(object):
    def __init__(self, model, embedding, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.embedding = embedding
        self.model.eval()

    def visualize(self):
        def compute_pw(output, sent1, sent2):
            grads1, grads2 = autograd.grad(torch.max(output), [sent1, sent2])
            grads1 = grads1[0].squeeze(0).squeeze(0).permute(1, 0)
            grads2 = grads2[0].squeeze(0).squeeze(0).permute(1, 0)
            grads1 = [torch.sqrt((g**2).sum()).cpu().data[0] for g in grads1]
            grads2 = [torch.sqrt((g**2).sum()).cpu().data[0] for g in grads2]
            return grads1, grads2

        def print_pw(pw, out, sent1_raw, sent2_raw):
            print("Prediction: {}".format(torch.max(out, 1)[1].cpu().data[0]))
            toks1 = sent1_raw.split()
            fmt_str = " ".join(["{:>%s}" % max(len(word), 6) for word in toks1])
            print(fmt_str.format(*toks1))
            max_val = np.max(pw[0])
            for word, val in zip(toks1, pw[0]):
                if abs(val) == max_val:
                    print("\x1b[1;33m", end="")
                print(("{:>%s}" % max(len(word), 6)).format(round(abs(val), 3)), end=" ")
                print("\x1b[1;0m", end="")
            print()

            toks2 = sent2_raw.split()
            fmt_str = " ".join(["{:>%s}" % max(len(word), 6) for word in toks2])
            print(fmt_str.format(*toks2))
            max_val = np.max(pw[1])
            for word, val in zip(toks2, pw[1]):
                if abs(val) == max_val:
                    print("\x1b[1;33m", end="")
                print(("{:>%s}" % max(len(word), 6)).format(round(abs(val), 3)), end=" ")
                print("\x1b[1;0m", end="")
            print("\n")

        def compute_corr_matrix(output, sent1, sent2):
            def cos_sim(v1, v2):
                return torch.dot(v1, v2) / (torch.sqrt(torch.sum(v1**2)) * torch.sqrt(torch.sum(v2**2)))
            grads1, grads2 = autograd.grad(torch.sum(output), [sent1, sent2])
            grads1 = grads1[0].squeeze(0).squeeze(0).permute(1, 0)
            grads2 = grads2[0].squeeze(0).squeeze(0).permute(1, 0)
            grads = torch.cat([grads1, grads2], 0)
            sz = grads.size(0)
            corr_matrix = np.empty((sz, sz))
            for i, g1 in enumerate(grads):
                for j, g2 in enumerate(grads):
                    corr_matrix[i, j] = torch.dot(g1, g2).cpu().data[0]
            return corr_matrix

        def print_matrices(corr_matrix, out, sent1_raw, sent2_raw):
            print("Prediction: {}".format(torch.max(out, 1)[1].cpu().data[0]))
            toks = sent1_raw.split()
            toks.extend(sent2_raw.split())
            max_len = max([len(tok) for tok in toks])
            fmt_str = " " * (max_len + 1) + " ".join(["{:>%s}" % max(len(word), 6) for word in toks])
            print(fmt_str.format(*toks))
            for i, (word, row) in enumerate(zip(toks, corr_matrix)):
                print(("{:>%s}" % max_len).format(word), end=" ")
                for j, (w2, val) in enumerate(zip(toks, row)):
                    if abs(val) > 0.1:
                        print("\x1b[1;33m", end="")
                    print(("{:>%s}" % max(len(w2), 6)).format(round(val, 3)), end=" ")
                    print("\x1b[1;0m", end="")
                print()

            s, v = np.linalg.eig(corr_matrix)
            fmt_str = " ".join(["{:>%s}" % max(len(word), 6) for word in toks])
            v = v.transpose()
            print(fmt_str.format(*toks))
            for row, s_val in zip(v, s):
                for word, val in zip(toks, row):
                    if abs(val) > 0.2:
                        print("\x1b[1;33m", end="")
                    print(("{:>%s}" % max(len(word), 6)).format(round(abs(val), 3)), end=" ")
                    print("\x1b[1;0m", end="")
                print(round(s_val, 3))
            print("============================")

        for batch in self.data_loader:
            for s1, s2, ext_feats, sr1, sr2, l in zip(batch.sentence_1.split(1, 0), batch.sentence_2.split(1, 0),
                    batch.ext_feats.split(1, 0), batch.sentence_1_raw, batch.sentence_2_raw, batch.label):
                if len(sr1.split()) + len(sr2.split()) < 20:
                    continue
                sent1 = self.embedding(s1).transpose(1, 2)
                sent2 = self.embedding(s2).transpose(1, 2)
                # sent2.requires_grad = sent1.requires_grad = True
                output = self.model(sent1, sent2)
                corr_matrix = compute_corr_matrix(output, sent1, sent2)
                print_matrices(corr_matrix, output, sr1, sr2)
                # pw = compute_pw(output, sent1, sent2)
                # print_pw(pw, output, sr1, sr2)

