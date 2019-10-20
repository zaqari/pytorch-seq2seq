import torch.nn as nn
import torch.nn.functional as F
import torch
from random import randint
import random
import numpy as np
import matplotlib.pyplot as plt

class model(nn.Module):

    def __init__(self, encoder, decoder, outlayer):
        super(model, self).__init__()
        self.enc = encoder
        self.dec = decoder
        self.out = outlayer
        self.one_hot_random_ = torch.eye(self.dec.n_classes)
        #self.parameters = list(self.enc.parameters())+list(self.dec.parameters())+list(self.atn.parameters())


    def analyze_sentence(self, encoder_data, decoder_data):
        """
        Full model utilization for forward. In this
         instance, mode=0 is .train() mode, mode=1 will be
         .eval() mode.
        For the moment, we're giving this a shot as the
         generator for the training period, too, though if
         need be we can switch it out to run the generator
         in line again.
        """

        encoder_outputs, hiddn = self.encode(encoder_data)
        hidden = [hiddn, hiddn]
        projected_keys = self.dec.attention.proj_layer(encoder_outputs)
        outputs, pre_outputs, alphas, hidden = self.decode(decoder_data, hidden, projected_keys, encoder_outputs)
        outputs = self.output(pre_outputs)

        return outputs, alphas


    def encode(self, encoder_data):
        encoder_outputs = []
        hidden = self.enc.initHidden()
        for word in range(len(encoder_data)):
            output, hidden = self.enc(encoder_data[word].view(-1), hidden)
            encoder_outputs.append(output)
        encoder_outputs = torch.cat(encoder_outputs, dim=1).squeeze(0)
        return encoder_outputs, hidden


    def decode(self, decoder_inputs, encoder_hidden_state, projected_keys, encoder_outputs):
        outputs, pre_outputs, alphas, hidden = self.dec(decoder_inputs, encoder_hidden_state, projected_keys, encoder_outputs)
        return outputs, pre_outputs, alphas, hidden


    def output(self, pre_outputs):
        return self.out(pre_outputs)


    def batched_training(self, enc_x, dec_x, y, optimizer, loss_criterion, validation_data=(), epochs=10, cutoff=.9, sample_pct=1.0, exploration_rate=None):

        for epoch in range(epochs):

            self.enc.train()
            self.dec.train()
            self.out.train()
            optimizer.zero_grad()

            print('Epoch {}/{}'.format(epoch+1, epochs))

            samples = list(np.random.choice(len(enc_x), int(len(enc_x) * sample_pct), replace=False).reshape(-1))
            for sent in samples:
                outputs, _ = self.analyze_sentence(enc_x[sent], dec_x[sent])

                loss = loss_criterion(outputs.view(-1, self.dec.n_classes), y[sent].view(-1))
                loss.backward()
                #self.grad_sanity_check()
                optimizer.step()
                optimizer.zero_grad()

            cut = self.epoch_statistics(validation_data=validation_data, train_data=(enc_x, dec_x, y), loss_criterion=loss_criterion)
            if cut >= cutoff:
                break


    def epoch_statistics(self, validation_data, train_data, loss_criterion):
        cut_off = None
        if bool(validation_data):
            cut_off = self.evaluation(validation_data[0], validation_data[1], validation_data[2], loss_criterion)
        else:
            cut_off = self.evaluation(train_data[0], train_data[1], train_data[2], loss_criterion)
        return cut_off


    def explore(self, output, randN, exploration_cutoff):
        out=output.detach()
        if randN > exploration_cutoff:
            return self.one_hot_random_[random.randint(0, self.dec.n_classes-1)]
        else:
            return out


    def evaluation(self, x, dec_x, Y, loss_criterion):

        self.enc.eval()
        self.dec.eval()
        self.out.eval()

        with torch.no_grad():
            accuracy, lossiness = [], []
            for i in range(len(x)):
                outputs, attention_data = self.analyze_sentence(x[i], dec_x[i])
                acc = (outputs.topk(1, dim=-1)[1].view(-1) == dec_x[i].view(-1)).sum().item() / len(dec_x[i])
                accuracy += [acc]
                lossiness += [loss_criterion(outputs.view(-1, self.dec.n_classes),
                                             Y[i].view(-1))]
                #[loss_criterion(outputs.squeeze(1), Y[i].view(-1))]
            print('@los: {:.4f} | @acc: {:.4f}'.format(sum(lossiness) / len(lossiness), sum(accuracy) / len(accuracy)))
            print('============] [============\n')
            return sum(accuracy) / len(accuracy)


    def grad_sanity_check(self):
        good = 0
        total = 0
        for name, param in self.named_parameters():
            if (param.grad is not None) and (param.requires_grad) and (torch.sum(param.grad.data).item() != 0):
                good += 1
            else:
                print(name)

            total += 1
        print(good, total)
        print()