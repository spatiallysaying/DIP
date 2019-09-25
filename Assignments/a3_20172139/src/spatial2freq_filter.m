
f=imread( 'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/lena.jpg');
% f=rgb2gray(img);

h=[0 1 0 ; 1 2 1;0 1 0]; %If h(x,y) is given in the spatial domain

% Padding
%Consider a matrix a with dimensions M × N and a matrix b with dimensions P × Q.
%Fill in matrices a, b by zeroes to have dimensions M + P ? 1, N + Q ? 1
PQ = paddedsize(size(f));
%convert into frequency domain
F = fft2(double(f), PQ(1), PQ(2));
H = fft2(double(h), PQ(1), PQ(2));

% Multiply complex Fourier spectra element-wise, C = A . ? B
F_fH = H.*F; 
% The result of the convolution c is obtained by the inverse Fourier transformation
ffi = ifft2(F_fH);
ffi = ffi(2:size(f,1)+1, 2:size(f,2)+1); %Remove padding

%Display results (show all values)
figure, imshow(ffi,[])

