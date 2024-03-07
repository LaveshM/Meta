% Parameters
mu_real = 0;         % Mean for real part
sigma_real = 1;      % Standard deviation for real part
mu_imag = 0;         % Mean for imaginary part
sigma_imag = 1;      % Standard deviation for imaginary part

% Generate a 10x10x10000 matrix with complex Gaussian random values
real_part = normrnd(mu_real, sigma_real, [2, 4, 10000])/1.414;
imag_part = normrnd(mu_imag, sigma_imag, [2, 4, 10000])/1.414;
H = complex(real_part, imag_part);

% Save the complex matrix to a file (e.g., 'complex_gaussian_matrix.mat')
save('Hrand.mat', 'H');

disp('Complex matrix saved successfully.');
