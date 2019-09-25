footBall=imread( 'C:/SAI/IIIT/2019_Monsoon/DIP/Assignment3/input_data/land.png');

imshow(footBall)

%Determine good padding for Fourier transform
PQ = paddedsize(size(footBall));

%Create Notch filters corresponding to extra peaks in the Fourier transform
H1 = notch('btw', PQ(1), PQ(2), 10, 50, 100);
H2 = notch('btw', PQ(1), PQ(2), 10, 1, 400);
H3 = notch('btw', PQ(1), PQ(2), 10, 620, 100);
H4 = notch('btw', PQ(1), PQ(2), 10, 22, 414);
H5 = notch('btw', PQ(1), PQ(2), 10, 592, 414);
H6 = notch('btw', PQ(1), PQ(2), 10, 1, 114);

% Calculate the discrete Fourier transform of the image
F=fft2(double(footBall),PQ(1),PQ(2));

% Apply the notch filters to the Fourier spectrum of the image
FS_football = F.*H1.*H2.*H3.*H4.*H5.*H6;

% convert the result to the spacial domain.
F_football=real(ifft2(FS_football)); 

% Crop the image to undo padding
F_football=F_football(1:size(footBall,1), 1:size(footBall,2));

%Display the blurred image
figure, imshow(F_football,[])

% Display the Fourier Spectrum 
% Move the origin of the transform to the center of the frequency rectangle.
Fc=fftshift(F);
Fcf=fftshift(FS_football);

% use abs to compute the magnitude and use log to brighten display
S1=log(1+abs(Fc)); 
S2=log(1+abs(Fcf));
figure, imshow(S1,[])
figure, imshow(S2,[])