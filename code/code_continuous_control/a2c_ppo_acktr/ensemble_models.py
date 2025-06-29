import torch, math
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# ensemble of linear layers parallelized for GPU
class EnsembleLinearGPU(nn.Module):
    def __init__(self, in_features, out_features, n_ensemble, bias=True, init_param = 'tanh'):
        super(EnsembleLinearGPU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_ensemble = n_ensemble
        self.bias = bias
        self.init_param = init_param
        self.weights = nn.Parameter(torch.Tensor(n_ensemble, out_features, in_features))
        if bias:
            self.biases  = nn.Parameter(torch.Tensor(n_ensemble, out_features))
        else:
            self.register_parameter('biases', None)
        self.reset_parameters()

    # #original initialization
    # def reset_parameters(self):
    #     for weight in self.weights:
    #         w = nn.Linear(self.in_features, self.out_features)
    #         torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    #     if self.biases is not None:
    #         for bias in self.biases:
    #             fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights[0])
    #             bound = 1 / math.sqrt(fan_in)
    #             torch.nn.init.uniform_(bias, -bound, bound)
    def reset_parameters(self):
        if self.init_param == 'tanh':
            gain = torch.nn.init.calculate_gain('tanh')  # Calculate gain factor for Tanh
        elif self.init_param == 'relu':
            gain = torch.nn.init.calculate_gain('relu')
        elif self.init_param == 'sqrt':
            gain = torch.sqrt(torch.tensor(2.0))
        elif self.init_param == 'zero':
            # Initialize both weights and biases to zero
            for weight in self.weights:
                torch.nn.init.zeros_(weight)
            if self.biases is not None:
                torch.nn.init.zeros_(self.biases)
            return  # Early return to skip
        else:
            gain = 1

        for weight in self.weights:
            torch.nn.init.orthogonal_(weight, gain=gain)  # Orthogonal initialization with gain
        if self.biases is not None:
            torch.nn.init.constant_(self.biases, 0)  # Initialize biases to 0

    def forward(self, inputs):
        # check input sizes
        if inputs.dim() == 3:
            # assuming size is [n_ensemble x batch_size x features]
            assert(inputs.size(0) == self.n_ensemble and inputs.size(2) == self.in_features)
        elif inputs.dim() == 2:
            n_samples, n_features = inputs.size(0), inputs.size(1)
            assert (n_samples % self.n_ensemble == 0 and n_features == self.in_features), [n_samples, self.n_ensemble, n_features, self.in_features]
            batch_size = int(n_samples / self.n_ensemble)
            inputs = inputs.view(self.n_ensemble, batch_size, n_features)

        # reshape to [n_ensemble x n_features x batch_size]
        inputs = inputs.permute(0, 2, 1)
        outputs = torch.bmm(self.weights, inputs)
        outputs = outputs
        if self.bias:
            outputs = outputs + self.biases.unsqueeze(2)
        # reshape to [n_ensemble x batch_size x n_features]
        outputs = outputs.permute(0, 2, 1).contiguous()
        return outputs


class Policy(nn.Module):
    def __init__(self, num_inputs, n_actions, n_hidden):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(num_inputs, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_actions + 1)

    def forward(self, obs):
        h = F.relu(self.layer1(obs))
        h = F.relu(self.layer2(h))
        h = self.layer3(h)
        a = h[:, :self.n_actions]
        v = h[:, -1]
        return a, v

class ValueNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, n_hidden):
        super(ValueNetwork, self).__init__()
        self.layer1 = nn.Linear(n_inputs, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, 1)

    def forward(self, obs):
        h = F.relu(self.layer1(obs))
        h = F.relu(self.layer2(h))
        v = self.layer3(h)
        return v

class PolicyEnsembleCNN(nn.Module):
    def __init__(self, num_inputs, n_actions, n_hidden, n_ensemble):
        super(PolicyEnsembleCNN, self).__init__()

        hidden_size = 512

        self.conv1 = nn.Conv2d(in_channels=num_inputs, out_channels=32, kernel_size=8,\
                      stride=4, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,\
                      stride=2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,\
                      stride=1, padding=0,bias=True)
        self.fc1 = nn.Linear(3136, hidden_size)
        self.relu = nn.ReLU()


        self.layer1 = EnsembleLinearGPU(hidden_size, n_hidden, n_ensemble)
        self.layer2 = EnsembleLinearGPU(n_hidden, n_hidden, n_ensemble)
        self.layer3 = EnsembleLinearGPU(n_hidden, n_actions, n_ensemble)

    def forward(self, obs):
        x = self.relu(self.conv1(obs/ 255.0))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        out = self.relu(self.fc1(x))

        h = F.relu(self.layer1(out))
        h = F.relu(self.layer2(h))
        a = self.layer3(h)
        return a




class PolicyEnsembleCNNDropout(nn.Module):
    def __init__(self, num_inputs, n_actions, n_hidden, p_dropout=0.1):
        super(PolicyEnsembleCNNDropout, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                 constant_(x, 0), nn.init.calculate_gain('relu'))

        hidden_size = 512
        self.p_dropout = p_dropout

        self.conv1 = nn.Conv2d(in_channels=num_inputs, out_channels=32, kernel_size=8,\
                      stride=4, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,\
                      stride=2, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,\
                      stride=1, padding=0,bias=True)
        self.fc1 = nn.Linear(3136, hidden_size)
        self.relu = nn.ReLU()


        self.layer1 = nn.Linear(hidden_size, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_actions)

    def forward(self, obs):
        x = self.relu(self.conv1(obs/ 255.0))
        x = F.dropout2d(x, p = self.p_dropout)
        x = self.relu(self.conv2(x))
        x = F.dropout2d(x, p = self.p_dropout)
        x = self.relu(self.conv3(x))
        x = F.dropout2d(x, p = self.p_dropout)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        out = self.relu(self.fc1(x))
        out = F.dropout(out, p = self.p_dropout)

        h = F.relu(self.layer1(out))
        h = F.dropout(h, p = self.p_dropout)
        h = F.relu(self.layer2(h))
        h = F.dropout(h, p = self.p_dropout)
        a = self.layer3(h)
        return a


class PolicyEnsembleMLP(nn.Module):
    def __init__(self, n_inputs, n_actions, n_hidden, n_ensemble):
        super(PolicyEnsembleMLP, self).__init__()

        self.layer1 = EnsembleLinearGPU(n_inputs, n_hidden, n_ensemble)
        self.layer2 = EnsembleLinearGPU(n_hidden, n_hidden, n_ensemble)
        self.layer3 = EnsembleLinearGPU(n_hidden, n_actions, n_ensemble)

    def forward(self, obs):
        h = F.relu(self.layer1(obs))
        h = F.relu(self.layer2(h))
        a = self.layer3(h)
        return a
    
# class PolicyEnsembleMLP_simple(nn.Module):
#     def __init__(self, n_inputs, n_actions, n_hidden, n_ensemble):
#         super(PolicyEnsembleMLP_simple, self).__init__()

#         self.layer1 = EnsembleLinearGPU(n_inputs, n_hidden, n_ensemble, init_param = 'sqrt')
#         self.layer2 = EnsembleLinearGPU(n_hidden, n_hidden, n_ensemble, init_param = 'sqrt')
#         self.layer3 = EnsembleLinearGPU(n_hidden, n_actions, n_ensemble, init_param = 'None')

#     def forward(self, obs):
#         h = torch.tanh(self.layer1(obs))
#         h = torch.tanh(self.layer2(h))
#         a = self.layer3(h)
#         return a

class PolicyEnsembleMLP_simple(nn.Module):
    def __init__(self, n_inputs, n_actions, n_hidden, n_ensemble):
        super(PolicyEnsembleMLP_simple, self).__init__()

        self.layer1 = EnsembleLinearGPU(n_inputs, n_hidden, n_ensemble, init_param = 'sqrt')
        self.layer2 = EnsembleLinearGPU(n_hidden, n_hidden, n_ensemble, init_param = 'sqrt')
        self.layer3 = EnsembleLinearGPU(n_hidden, n_actions, n_ensemble, init_param = 'None')

    def forward(self, obs):
        h = torch.tanh(self.layer1(obs))
        h = torch.tanh(self.layer2(h))
        a = self.layer3(h)
        return a

class PolicyEnsembleMLP_nonrealizable(nn.Module):
    def __init__(self, n_inputs, n_actions, n_hidden, n_ensemble):
        super(PolicyEnsembleMLP_nonrealizable, self).__init__()

        self.layer1 = EnsembleLinearGPU(n_inputs, n_hidden, n_ensemble, init_param = 'sqrt')
        self.layer3 = EnsembleLinearGPU(n_hidden, n_actions, n_ensemble, init_param = 'None')

    def forward(self, obs):
        h = torch.tanh(self.layer1(obs))
        a = self.layer3(h)
        return a
    

class PolicyEnsembleMLP_linear(nn.Module):
    def __init__(self, n_inputs, n_actions,n_hidden,n_ensemble):
        super(PolicyEnsembleMLP_linear, self).__init__()

        self.layer1 = EnsembleLinearGPU(n_inputs, n_actions, n_ensemble, init_param = 'sqrt')

    def forward(self, obs):
        a = self.layer1(obs)
        return a


def custom_activation(x):
    pos_part = torch.log(1 + x + 0.5 * x ** 2)
    neg_part = -torch.log(1 - x + 0.5 * x ** 2)
    # Use torch.where to combine these based on the condition x > 0
    result = torch.where(x > 0, pos_part, neg_part)
    return result

class PolicyEnsembleMLP_simple_test(nn.Module):
    def __init__(self, n_inputs, n_actions, n_hidden, n_ensemble):
        super(PolicyEnsembleMLP_simple_test, self).__init__()

        self.layer1 = EnsembleLinearGPU(n_inputs, n_hidden, n_ensemble, init_param = 'sqrt')
        self.layer2 = EnsembleLinearGPU(n_hidden, n_hidden, n_ensemble, init_param = 'sqrt')
        self.layer3 = EnsembleLinearGPU(n_hidden, n_actions, n_ensemble, init_param = 'None')

    def forward(self, obs):
        h = custom_activation(self.layer1(obs))
        h = custom_activation(self.layer2(h))
        a = self.layer3(h)
        return a
    


class PolicyEnsembleMLP_Gaussian(nn.Module):
    def __init__(self, n_inputs, n_actions, n_hidden, n_ensemble):
        super(PolicyEnsembleMLP_Gaussian, self).__init__()

        self.layer1 = EnsembleLinearGPU(n_inputs, n_hidden, n_ensemble, init_param='sqrt')
        self.layer2 = EnsembleLinearGPU(n_hidden, n_hidden, n_ensemble, init_param='sqrt')
        self.mean_layer = EnsembleLinearGPU(n_hidden, n_actions, n_ensemble, init_param='None')
        self.log_std_param = nn.Parameter(torch.full((n_ensemble, n_actions), -1.0))
        #self.log_std_param nn.Parameter(torch.zeros(n_ensemble, n_actions))  # Learnable log std

    def forward(self, obs):
        h = torch.tanh(self.layer1(obs))
        h = torch.tanh(self.layer2(h))
        mean = self.mean_layer(h)
        log_std = self.log_std_param.unsqueeze(1).expand_as(mean)
        return mean, log_std




class PolicyEnsembleDuckieTownCNN(nn.Module):
    def __init__(self, num_inputs, n_actions, n_hidden, n_ensemble):
        super(PolicyEnsembleDuckieTownCNN, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                 constant_(x, 0), nn.init.calculate_gain('relu'))

        hidden_size = 512
        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)
        self.lin1 = nn.Linear(flat_size, hidden_size)

        self.layer1 = EnsembleLinearGPU(hidden_size, n_hidden, n_ensemble)
        self.layer2 = EnsembleLinearGPU(n_hidden, n_hidden, n_ensemble)
        self.layer3 = EnsembleLinearGPU(n_hidden, n_actions, n_ensemble)

    def forward(self, obs):
        x = obs
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        out = self.lr(self.lin1(x))

        h = F.relu(self.layer1(out))
        h = F.relu(self.layer2(h))
        a = self.layer3(h)
        return a


