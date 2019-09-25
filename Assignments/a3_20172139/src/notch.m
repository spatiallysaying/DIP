function H = notch(type, M, N, D0, x, y, n)
%notch Computes frequency domain notch filters
%   H = NOTCH(TYPE, M, N, D0, x, y, n) creates the transfer function of
%   a notch filter, H, of the specified TYPE and size (M-by-N). centered at
%   Column X, Row Y in an unshifted Fourier spectrum.
%   Valid values for TYPE, D0, and n are:
%
%   'ideal'     Ideal highpass filter with cutoff frequency D0.  n
%               need not be supplied.  D0 must be positive
%
%   'btw'       Butterworth highpass filter of order n, and cutoff D0.
%               The default value for n is 1.0.  D0 must be positive.
%
%   'gaussian'  Gaussian highpass filter with cutoff (standard deviation)
%               D0.  n need not be supplied.  D0 must be positive.
% 

% The transfer function Hhp of a highpass filter is 1 - Hlp,
% where Hlp is the transfer function of the corresponding lowpass
% filter.  Thus, we can use function lpfilter to generate highpass
% filters.

if nargin == 6
   n = 1; % Default value of n.
end

% Generate highpass filter.
Hlp = lpfilter(type, M, N, D0, n);
H = 1 - Hlp;
H = circshift(H, [y-1 x-1]);
