import torch
import typing
import pickle
import numpy as np
from scipy.linalg import sqrtm, dft
from numpy.random import default_rng
import scipy.io as sio
Bern = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.5]))  # equal prob.

class SineDataset(torch.utils.data.Dataset):
    """generate data of a random sinusoidal task
    y = A sin(x + phi) + epsilon,
    where: epsilon is sampled from N(0, s^2)
    """
    def __init__(
        self,
        amplitude_range: typing.Tuple[float],
        phase_range: typing.Tuple[float],
        noise_std: float,
        x_range: typing.Tuple[float],
        num_samples: int
    ) -> None:
        """
        Args:
            amplitudes: a tuple consisting the range of A
            phase: a tuple consisting the range of phase
            noise_variance: the variance of the noise
        """
        super().__init__()
        
        self.amplitude_range = [a for a in amplitude_range]
        self.phase_range = [phi for phi in phase_range]
        self.noise_std = noise_std

        self.x = torch.linspace(start=x_range[0], end=x_range[1], steps=num_samples)

    def __len__(self) -> int:
        return 100000

    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        """generate data of a task
        """
        y = self.generate_label()
        y = y + torch.randn_like(input=y) * self.noise_std

        return [self.x, y]
    
    def generate_label(self) -> torch.Tensor:
        """
        """
        # sample parameters randomly
        amplitude = torch.rand(size=(1,)) * (self.amplitude_range[1] - self.amplitude_range[0]) + self.amplitude_range[0]
        phase = torch.rand(size=(1,)) * (self.phase_range[1] - self.phase_range[0]) + self.phase_range[0]

        y = amplitude * torch.sin(input=self.x + phase)

        return y

class LineDataset(torch.utils.data.Dataset):
    """generate a data for a task following the formula:
    y = ax + b
    """
    def __init__(self, slope_range: typing.Tuple[float], intercept_range: typing.Tuple[float], x_range: typing.Tuple[float], num_samples: int, noise_std: float) -> None:
        super().__init__()
        
        self.slope_range = [a for a in slope_range]
        self.intercept_range = [b for b in intercept_range]
        self.noise_std = noise_std

        self.x = torch.linspace(start=x_range[0], end=x_range[1], steps=num_samples)

    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        y = self.generate_label()

        y = y + torch.randn_like(input=y) * self.noise_std

        return [self.x, y]

    def __len__(self) -> int:
        return 100000

    def generate_label(self) -> torch.Tensor:
        slope = torch.rand(size=(1,)) * (self.slope_range[1] - self.slope_range[0]) + self.slope_range[0]
        intercept = torch.rand(size=(1,)) * (self.intercept_range[1] - self.intercept_range[0]) + self.intercept_range[0]

        y = slope * self.x + intercept

        return y
    
class Demodulator(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open('./offline_realistic/meta_train_set/input_set.pckl', 'rb') as file:
            data = pickle.load(file)
        i = torch.randint(0,1000,())
        
        self.x = data[i][:,2:4]
        
        labels = data[i][:,0:2]
        
        self.y = self.encode_labels(labels[:,0],labels[:,1])
        print(self.x.shape,self.y.shape)
        
        

    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        return [self.x, self.y]

    def __len__(self) -> int:
        return 100000
    
    def encode_labels(self, a: int, b: int) -> int:
        """Encode pair (a, b) into a unique numerical label"""
        mapping = {(-3, -3): 0, (-3, -1): 1, (-3, 1): 2, (-3, 3): 3,
                (-1, -3): 4, (-1, -1): 5, (-1, 1): 6, (-1, 3): 7,
                (1, -3): 8, (1, -1): 9, (1, 1): 10, (1, 3): 11,
                (3, -3): 12, (3, -1): 13, (3, 1): 14, (3, 3): 15}
        
        pairs = torch.stack((a, b), dim=1)
        encoded_labels = torch.tensor([mapping[tuple(pair.tolist())] for pair in pairs])

        return encoded_labels


class TDemodulator(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.labels = torch.arange(1000)%4
        self.x = torch.zeros((1000,2))
        #x will be from 
        self.x[:,0] = torch.tile(torch.FloatTensor([-1,-1,1,1]), (250,1)).reshape(-1)
        self.x[:,1] = torch.tile(torch.FloatTensor([-1,1,-1,1]), (250,1)).reshape(-1)
        #divide by root 2
        self.x = self.x/np.sqrt(2)
        self.fl = True
        
        
    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        #h = torch.FloatTensor([torch.randint(0, 2, ()) * 2 - 1,0])
        #sample randomly from 4 numbers
        t = torch.FloatTensor([torch.randint(0, 4, ()) * 0.2 +0.2])
        #cov matix is 1,t,t,1
        cov = torch.FloatTensor([[1,t],[t,1]])
        #h is a 2x1 vector sampled from a complex normal distribution
        [h1,h2] = torch.FloatTensor(np.random.multivariate_normal(mean=np.zeros(2), cov=cov, size=2))
        #y1 is complex multiplication of h1 and x
        
        self.y1 = torch.zeros((1000,2))
        noise = torch.FloatTensor(np.random.multivariate_normal(mean=np.zeros(2), cov=0.1 * np.eye(2), size=1000))
        self.y1[:, 0] = h1[0] * self.x[:,0] - h1[1] * self.x[:,1] + noise[:,0]
        self.y1[:, 1] = h1[1] * self.x[:,0] + h1[0] * self.x[:,1] + noise[:,1]
        #similarly y2 with h2
        self.y2 = torch.zeros((1000,2)) 
        noise = torch.FloatTensor(np.random.multivariate_normal(mean=np.zeros(2), cov=0.1 * np.eye(2), size=1000))
        self.y2[:, 0] = h2[0] * self.x[:,0] - h2[1] * self.x[:,1] + noise[:,0]
        self.y2[:, 1] = h2[1] * self.x[:,0] + h2[0] * self.x[:,1] + noise[:,1]
        
        self.y = torch.zeros((1000,4))
        self.y[:,:2] = self.y1
        self.y[:,2:] = self.y2
        
        
        # self.x = torch.zeros((1000,2))
        # noise = torch.FloatTensor(np.random.multivariate_normal(mean=np.zeros(2), cov=0.1 * np.eye(2), size=1000))
        # self.x[:, 0] = h[0] * self.labels[:,0] + h[1] * self.labels[:,1] + noise[:,0]
        # self.x[:, 1] = noise[:,1]
        #print(self.x.shape,self.y.shape)
        
        if self.fl:
            print(self.y[1:3,:])
            print(self.x[1:3,:])
            print(self.labels[1:3])
            print(h1,h2)
            self.fl = False
        return [self.y, self.labels]

    def __len__(self) -> int:
        return 100000
    
    def encode_labels(self, a: int, b: int) -> int:
        """Encode pair (a, b) into a unique numerical label"""
        mapping = {(-3,0): 0, (-1,0): 1, (1,0): 2, (3,0): 3,}
        
        pairs = torch.stack((a, b), dim=1)
        encoded_labels = torch.tensor([mapping[tuple(pair.tolist())] for pair in pairs])

        return encoded_labels
    

class KDemodulator(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.y = torch.zeros((1000,2))
        self.y[:,0] = torch.arange(1000)%4
        self.y[:,1] = torch.arange(1000)%4
        self.labels = torch.zeros((1000,2,2))
        self.labels[:,0,0] = torch.tile(torch.FloatTensor([-3,-1,1,3]), (250,1)).reshape(-1)
        self.labels[:,1,0] = torch.tile(torch.FloatTensor([-3,-1,1,3]), (250,1)).reshape(-1)
    
        
    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        h1 = torch.FloatTensor([torch.randint(0, 2, ()) * 2 - 1,0])
        h2 = torch.FloatTensor([torch.randint(0, 2, ()) * 2 - 1,0])
        # print(h)
        # h = torch.FloatTensor([-1,0])
        
        self.x = torch.zeros((1000,2))
        noise = torch.FloatTensor(np.random.multivariate_normal(mean=np.zeros(2), cov=0.1 * np.eye(2), size=1000))
        self.x[:, 0] = h1[0] * self.labels[:,0,0] + h1[1] * self.labels[:,0,1]+ h2[0] * self.labels[:,1,0] + h2[1] * self.labels[:,1,1]+ noise[:,0]
        #self.x[:, 1] = h1[0] * self.labels[:,0,0] + h1[1] * self.labels[:,0,1]+ h2[0] * self.labels[:,1,0] + h2[1] * self.labels[:,1,1]+ noise[:,1]
        self.x[:, 1] = noise[:,1]
        return [self.x, self.y]

    def __len__(self) -> int:
        return 100000
    
    def encode_labels(self, a: int, b: int) -> int:
        """Encode pair (a, b) into a unique numerical label"""
        mapping = {(-3,0): 0, (-1,0): 1, (1,0): 2, (3,0): 3,}
        
        pairs = torch.stack((a, b), dim=1)
        encoded_labels = torch.tensor([mapping[tuple(pair.tolist())] for pair in pairs])

        return encoded_labels
    

# class KKDemodulator(torch.utils.data.Dataset):
#     def __init__(self) -> None:
#         super().__init__()
#         self.k=2
#         self.N = 3200
#         self.y = torch.zeros((self.N,self.k))
#         for k in range(k):
#             self.y[:,k] = torch.arange(self.N)%4
        
#         self.p = torch.zeros(self.N, dtype=complex)
#         self.p.real = torch.tile(torch.FloatTensor([-1,-1,1,1]), (250,1)).reshape(-1)
#         self.p.imag = torch.tile(torch.FloatTensor([-1,1,-1,1]), (250,1)).reshape(-1)
#         #identity matrix
#         self.R_uniform = 0.8 * torch.eye(self.k) + 0.3 * torch.ones(self.k,self.k)
        
    
        
#     def __getitem__(self, index) -> typing.List[torch.Tensor]:
#         h1 = torch.complex(torch.randn(()), torch.randn(()))
#         h2 = torch.complex(torch.randn(()), torch.randn(()))
        
#         self.x = torch.zeros((1000, 2))
#         noise = torch.FloatTensor(np.random.multivariate_normal(mean=np.zeros(2), cov=0.1 * np.eye(2), size=1000))
#         self.x[:, 0] = h1.real * self.labels[:, 0, 0] + h1.imag * self.labels[:, 0, 1] + h2.real * self.labels[:, 1, 0] + h2.imag * self.labels[:, 1, 1] + noise[:, 0]
#         self.x[:, 1] = noise[:, 1]
#         return [self.x, self.y]

#     def __len__(self) -> int:
#         return 100000
    
#     def channel_generation_correlated(Ml, N, R_uniform, lsf):
#         rng = default_rng()
#         Gamma = np.zeros((N, Ml), dtype=complex)
#         Rg = np.zeros((Ml, Ml, N), dtype=complex)
#         Rh = np.zeros((Ml, Ml, N), dtype=complex)
#         H = np.zeros((Ml, N), dtype=complex)

#         Ar = 1 / np.sqrt(Ml) * dft(Ml)

#         for k in range(N):
#             Gamma[k, :] = lsf[k]
#             Rg[:, :, k] = np.diag(np.sqrt(Gamma[k, :])) @ R_uniform @ np.diag(np.sqrt(Gamma[k, :]))
#             Rh[:, :, k] = Ar @ Rg[:, :, k] @ Ar.conj().T
#             H[:, k] = np.squeeze(sqrtm(Rh[:, :, k]) @ (rng.standard_normal((Ml, 1)) + 1j * rng.standard_normal((Ml, 1))) / np.sqrt(2))

#         # Ha = Ar.conj().T @ H

#         return H
    
class KKDemodulator(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        HH = sio.loadmat('H5.mat')['H']
        self.H = torch.tensor(HH, dtype=torch.complex64)
        self.Ml = self.H.shape[0]
        self.k = self.H.shape[1]
        self.T = self.H.shape[2]
        self.N = 3200
        self.y = torch.zeros(self.N)
        
        self.y = torch.arange(self.N)%4
        
        self.p = torch.zeros(self.N, dtype=torch.complex64)
        self.p.real = torch.tile(torch.FloatTensor([-1,-1,1,1]), (self.N//4,1)).reshape(-1)
        self.p.imag = torch.tile(torch.FloatTensor([-1,1,-1,1]), (self.N//4,1)).reshape(-1)
        #identity matrix
        fll = True
        if fll:
            print('H',self.H.shape)
            print('p',self.p.shape)
            
            fll = False
        self.t = torch.randint(0,self.T,(1,))
        
            
        
    def __getitem__(self, index) -> typing.List[torch.Tensor]:
        self.t = (self.t+1)%10000
        h = torch.squeeze(self.H[:,:,self.t]).T
        # change p vector to a random permutaion of itself
        perm = torch.zeros( self.N, dtype=torch.int)
        X = torch.zeros( self.N, self.k, dtype=torch.complex64)
        labels = torch.zeros(self.N, self.k, dtype=torch.int)
        for i in range(self.k):
            perm = torch.randperm(self.N)
            X[:,i] = self.p[perm]
            labels[:,i] = self.y[perm]
        #sample noise from a complex normal
        
        noise = torch.randn(self.N, self.Ml, dtype=torch.complex64)
        res = X@h + 0.01* noise
        #break res complex into real and imaginary parts
        x = torch.zeros(self.N, 2*self.Ml)   
        x[:,0:self.Ml] = res.real
        x[:,self.Ml:] = res.imag
        
        
        return [x, labels]

    def __len__(self) -> int:
        return 100000
    
    