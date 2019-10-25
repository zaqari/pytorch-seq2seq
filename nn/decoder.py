import torch
import torch.nn as nn
import torch.nn.functional as F

class dec(nn.Module):

    def __init__(self, device, attention, embeddings, hidden_size=300, context_size=600, dropout_p=.1, bidirectional=True):
        super(dec, self).__init__()

        self.device = device
        self.n_layers = 2
        self.hidden_size = hidden_size
        self.n_classes = len(embeddings) + 1
        self.bi = bidirectional
        self.output_size = self.hidden_size

        self.bi_mul = 1
        if self.bi:
            self.bi_mul = 2

        #Embedding layer
        self.embedding = torch.FloatTensor(embeddings)
        embed_size = self.embedding.shape[-1]
        self.embedding = nn.Embedding.from_pretrained(self.embedding)

        #Layers
        self.attention = attention
        self.dropout = nn.Dropout(p=dropout_p)
        self.gru1 = nn.GRU(embed_size + context_size,
                           self.hidden_size,
                           bidirectional=self.bi)
        self.gru2 = nn.GRU(self.hidden_size * self.bi_mul + context_size,
                           self.hidden_size,
                           bidirectional=self.bi)
        self.pre_output_layer = nn.Linear(hidden_size + context_size + embed_size, hidden_size)
        self.relu = nn.LeakyReLU()

    def lexical_step(self, input, hidden, projected_keys, encoder_outputs):
        """
        The following is a step-wise function that takes some input,
        the projected versions of the encoder states (states are reduced
        to hidden_size-D using the projection layer prima facie), the
        hidden state from the prior step, and the actual encoder_outputs
        to create a representation of the word going into the decoder
        and its context at t=sentence.index(word)
        """
        hidden_states = []

        #(1) Convert input to an embedding.
        embedded = self.embedding(input).view(1,1,-1)
        #(2) Generate the context vector and alphas using the attention
        #    layer provided to the network on initialization.
        context, alphas = self.attention(hidden, projected_keys, encoder_outputs)

        # (3) GRU-1 Ops
        # (3.1) Concatenate the embedding rep of the word with the
        #       generated context vector.
        gru1_input = torch.cat([embedded, context], dim=-1)
        # (3.2) Pass the previous concatenation as input to the
        #       GRU-1 layer
        output, hidN = self.gru1(gru1_input, hidden[0])
        # (3.3) Apply relu function and dropout to output, in order to
        #       avoid disappearing gradients and overfiting,
        #       respectively.
        output = self.dropout(self.relu(output))
        # (3.4) Append hidden state to the hidden states index.
        hidden_states.append(hidN)

        # (4) GRU-2 Ops:
        #     GRU-2 is unique in comparison to -1, but the operations
        #     are mirrored.
        gru2_input = torch.cat([output, context], dim=-1)
        output, hidN = self.gru2(gru2_input, hidden[-1])
        output = self.dropout(self.relu(output))
        hidden_states.append(hidN)

        # (5) Output Manipulation
        # (5.1) Concatenate embedding, output, and context vectors
        #       to be passed to the softmax, output layer. This gives
        #       our output access to all information pertaining to
        #       this time-step.
        pre_output = torch.cat([embedded, output, context], dim=-1)
        # (5.2) Optionally, apply dropout to (5.1).
        pre_output = self.dropout(pre_output)
        # (5.3) Now, compress this information to hidden_size, and
        #       simultaneously allow for a grad and differentiation to
        #       be applied here!
        pre_output = self.relu(self.pre_output_layer(pre_output))

        return output, hidden_states, pre_output, alphas


    def forward(self, inputs, encoder_hidden_state, projected_keys, encoder_outputs, teacher_forcing=True):
        """
        The following takes in the entirety of the inputs for the
        decoder, passes it to our stepwise function, lexical_step(),
        and iterates through the sentence to create a useful output.
        """
        hidden, outputs, alphas_out, pre_outputs = encoder_hidden_state, [], [], []

        decoder_input=torch.tensor(0)
        for i in range(len(inputs)):
            output, hidden, pre_output, alphas = self.lexical_step(decoder_input, hidden, projected_keys, encoder_outputs)
            outputs.append(output)
            pre_outputs.append(pre_output)
            alphas_out.append(alphas)

            if teacher_forcing:
                decoder_input = inputs[i].view(-1)
            else:
                decoder_input = output.topk(1, dim=-1)[1].view(-1)

        return torch.cat(outputs, dim=1), torch.cat(pre_outputs, dim=1), alphas_out, hidden


    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)