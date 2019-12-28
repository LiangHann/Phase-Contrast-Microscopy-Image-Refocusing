clc;
close all;
clear all;

addpath(genpath('image'));
addpath(genpath('whyte_code'));
addpath(genpath('cho_code'));

% parameters
opts.prescale = 1; %%downsampling
opts.xk_iter = 5; %% the iterations of pruning isolated noise in kernel
opts.gamma_correct = 1.0;
opts.k_thresh = 0;
width = 25; % width for the Gaussian blur filter
opts.kernel_size = width;  saturation = 0;
lambda_pixel = 4e-3; lambda_grad = 4e-3;%opts.gamma_correct = 2.2;
lambda_tv = 0.002; lambda_l0 = 2e-4; weight_ring = 1;
sigma = 1; % sigma of the Gaussian blur kernel
lambda = 2e-4;
gamma = 5e-4;
kappa = 2;

y = double(imread('Low1.tif')); % read in the original image
% y = y(2*end/5+1:4*end/5, 3*end/4+1:end, 1); % take the top left quarter of the original image
y = y(491:750, 3*end/4+1:end, 1); % take the top left quarter of the original image
y = y/(max(y(:)));

figure(1);
imshow(y, []);
title('original input image');

kernel_blur = fspecial('gaussian', [width width], sigma);
y_blur = imfilter(y, kernel_blur, 'same', 'replicate');

figure(2);
imshow(kernel_blur, []);
title('blur kernel');

figure(3);
imshow(y_blur, []);
title('blurred image');

% y_blur = y;

%---------------------------------------------------------------------------------------------------------------------------------------------------
% % Get the kernel of the Phase Contrast image from MICCAI 2010 of Yin
% [nrows, ncols] = size(y);
% Rwid = 4000; Wwid = 800; MRadius = 3;
% [~, airykernel] = getPhaseImagingModelHAiry(nrows, ncols, Rwid, Wwid, MRadius);
%---------------------------------------------------------------------------------------------------------------------------------------------------
%% kernel estimation
if size(y_blur,3)==3
    yg = im2double(rgb2gray(y_blur));
else
    yg = im2double(y_blur);
end
tic;
[kernel, interim_latent] = blind_deconv(yg, lambda_pixel, lambda_grad, opts);
toc;
y_blur = im2double(y_blur);

%==============================================================
%% deblur with different method
[H, W] = size(y_blur);
y_w = wrap_boundary_liu(y_blur, opt_fft_size([H W]+size(kernel)-1));
y_tmp = y_w(1:H,1:W,:);

Latent1 = deconvSps(y_tmp, kernel, 1e-4, 200); % deblur with hyper-Laplacian prior

Latent2 = L0Deblur_dark_chanel(y_tmp, kernel, 1e-4, 5e-5, 2.0); % deblur with the internal image

Latent3 = deconv_dark_chanel(y_tmp, kernel, 1e-3); % deblur without sparse prior of gradient

Latent4 = L0Restoration(y_tmp, kernel, 4e-4); % deblur without sparse prior of dark channel


figure(4);
imshow(Latent1, []);
title('hyper-Laplacian');

figure(5);
imshow(Latent2, []);
title('internal image');

figure(6);
imshow(Latent3, []);
title('without SPG');

figure(7);
imshow(Latent4, []);
title('without SPDC');

imwrite(y, sprintf('C:/Users/lh248/Downloads/downloads/Data/DICPC/DePC/original.png'));
imwrite(Latent1, sprintf('C:/Users/lh248/Downloads/downloads/Data/DICPC/DePC/HyperLaplacian.png'));
imwrite(Latent2, sprintf('C:/Users/lh248/Downloads/downloads/Data/DICPC/DePC/InternalImage.png'));
imwrite(Latent3, sprintf('C:/Users/lh248/Downloads/downloads/Data/DICPC/DePC/withoutSPG.png'));
imwrite(Latent4, sprintf('C:/Users/lh248/Downloads/downloads/Data/DICPC/DePC/withoutSPDC.png'));


 %% denoise the deblurred image
 lambda = 3e-4;
 [denoise, artifact] = L0Dehalo_2(Latent1, airykernel, lambda, gamma, kappa);

 figure(8);
 imshow(denoise, []);
 title('denoised image');
 
 figure(9);
 imshow(artifact, []);
 title('artifacts image');

% %% deblur the deblurred image without denoising it first
% % estimate blur kernel
% deLatent1 = im2double(Latent1);
% tic;
% [kernel2, interim_latent2] = blind_deconv(deLatent1, lambda_pixel, lambda_grad, opts);
% toc;
% 
% % deblur the image
% [H, W] = size(deLatent1);
% deLatent1_w = wrap_boundary_liu(deLatent1, opt_fft_size([H W]+size(kernel2)-1));
% deLatent1_tmp = deLatent1_w(1:H,1:W,:);
% dedeblur = deconvSps(deLatent1_tmp, kernel2, 1e-5, 200); % deblur with hyper-Laplacian prior
% dedeblur1 = ringing_artifacts_removal_new(deLatent1_tmp, kernel2, weight_ring); % deblur with ringing artifacts removal method
% dedeblur2 = L0Deblur_dark_chanel(deLatent1_tmp, kernel2, 1e-3, 5e-5, 2.0); % deblur with the internal image
% 
% dedeblur(dedeblur<0) = 0;
% dedeblur(dedeblur>1) = 1;
% 
% figure(9);
% imshow(dedeblur, []);
% title('deblur the deblurred image');
% 
% %% deblur the deblurred image with denoising it first
% % estimate blur kernel
% denoise = im2double(denoise);
% tic;
% [kernel3, interim_latent3] = blind_deconv(denoise, lambda_pixel, lambda_grad, opts);
% toc;
% 
% % deblur the image
% [H, W] = size(denoise);
% denoise_w = wrap_boundary_liu(denoise, opt_fft_size([H W]+size(kernel3)-1));
% denoise_tmp = denoise_w(1:H,1:W,:);
% dedenoise = deconvSps(denoise_tmp, kernel3, 1e-1, 200); % deblur with hyper-Laplacian prior
% dedenoise1 = ringing_artifacts_removal_new(denoise_tmp, kernel3, weight_ring); % deblur with ringing artifacts removal method
% dedenoise2 = L0Deblur_dark_chanel(denoise_tmp, kernel3, 1e-3, 5e-5, 2.0); % deblur with the internal image
% 
% % dedenoise(dedenoise<0) = 0;
% % dedenoise(dedenoise>1) = 1;
% 
% figure(10);
% imshow(dedenoise, [0, 1]);
% title('deblur the denoised image');
