import torch
import torch.nn as nn
import torch.nn.functional as F

class bahdanau(nn.Module):

    def __init__(self, key_size, query_size, hidden_size):
        super(bahdanau, self).__init__()
        self.hidden_size = hidden_size
        self.proj_layer = nn.Linear(key_size, hidden_size)
        self.query_layer = nn.Linear(query_size, hidden_size)
        #self.reduction_layer = nn.Linear(hidden_size*2, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.alphas = None

    def forward(self, hidden, keys, encoder_outputs):
        """ This function lives and dies off of having previously
            calculated the projection layer . . . this is run in
            training outside of the forward() call, post creation of
            the encoder outputs. """

        #(1) Construct query vector after flattenting hidden states
        # (1.1) Flatten hidden states
        query_input = torch.cat(hidden, dim=-1).view(-1)

        # (1.2) pass to the query layer NN realization
        query = self.query_layer(query_input)

        # (1.3) Since we have variable queryN v. encoder_ouputN
        #       (we're masking NOTHING!), we need to blow up the
        #       the number of query repetitions by the number of
        #       encoder outputs for the energy_layer.
        #query = query.repeat(encoder_outputs.size()[0]).view(-1, self.hidden_size)

        #(2) Take the encoder_outputs and create a key_layer
        ##### THIS MAY NEED TO BE MOVED OUTSIDE OF THE LOOP
        #keys = self.key_layer(encoder_outputs)

        #(3) Combine query & key, and then pass those to energy_layer.
        alphas = self.softmax(self.energy_layer(torch.tanh(query + keys))).transpose(0, -1).unsqueeze(0)

        #(4) Multiply and sum everything.
        values = encoder_outputs.unsqueeze(0) #torch.transpose(encoder_outputs, -1, 0).unsqueeze(0)
        context = torch.bmm(alphas, values)
        context = context.view(1, 1, -1) #torch.transpose(context.squeeze(0), -1, 0).view(1, 1, -1)

        return context, alphas

class dot(nn.Module):

    def __init__(self, key_size, query_size, hidden_size):
        super(dot, self).__init__()
        #self.hidden_size = hidden_size
        #self.key_size = key_size
        #self.query_size = query_size
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        #self.l3 = nn.Linear(hidden_size, hidden_size)
        #self.alpha_layer = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        #self.tanh = nn.Tanh()

        #PROJECT THIS LAYER OUTSIDE OF THE FORWARD LOOP . . . .
        # AS IN, CALL IT OVER ENCODER OUTPUTS PRIOR TO FORWARD
        #    proj_keys = attention.proj_layer(encoder_outputs)
        #    context, alphas = attention(hidden_, proj_keys)
        # THiS IS PROBABLY TO SAVE THE NUMBER OF TIMES THIS
        # GETS WEIGHTED/RECALCULATED.
        self.proj_layer = nn.Linear(key_size, hidden_size)

        #self.alphas = None

    def forward(self, hidden, projections, encoder_outputs):
        """ This function lives and dies off of having previously
         calculated the projection layer . . . this is run in
         training outside of the forward() call, post creation of
         the encoder outputs. """

        #(1) Construct query vector after flattenting hidden states
        # (1.1) Flatten hidden states
        query_input = sum(hidden)
        alphas = self.softmax(self.l1(torch.bmm(projections.unsqueeze(0), query_input.transpose(1, -1)))).transpose(1, -1)

        context = self.softmax(self.l2(torch.bmm(alphas, encoder_outputs.unsqueeze(0))))
        #context = self.relu(self.l1(context.squeeze(0)))
        #context = self.relu(self.l2(context))
        #context = self.relu(self.l3(context))

        """
        # (1.2) pass to the query layer NN realization
        query = self.query_layer(query_input)

        # (1.3) Since we have variable queryN v. encoder_ouputN
        #       (we're masking NOTHING!), we need to blow up the
        #       the number of query repetitions by the number of
        #       encoder outputs for the energy_layer.
        #query = query.repeat(encoder_outputs.size()[0]).view(-1, self.hidden_size)

        #(2) Take the encoder_outputs and create a key_layer
        keys = self.key_layer(encoder_outputs)

        #(3) Combine query & key, and then pass those to energy_layer.
        alphas = self.softmax(self.energy_layer(torch.tanh(query + keys))).transpose(0, -1).unsqueeze(0)

        #(4) Multiply and sum everything.
        values = encoder_outputs.unsqueeze(0) #torch.transpose(encoder_outputs, -1, 0).unsqueeze(0)
        context = torch.bmm(alphas, values)
        context = context.view(1, 1, -1) #torch.transpose(context.squeeze(0), -1, 0).view(1, 1, -1)
        """
        return context, alphas