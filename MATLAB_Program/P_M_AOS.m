%% This program implement regularized P_M equation by semi-implicit schema
%% with AOS algorithm.It will call gauss() for calculation of |grad(I_sigama)|
%% and Thomas() to solve a tri-diagonal liniear equation.

function energy = calculateEnergy(I, g)
    [Ix, Iy] = gradient(I);
    gradMag = sqrt(Ix.^2 + Iy.^2);
    energy = sum(sum(g .* gradMag.^2)); % 简化的能量计算公式
end


clear all;
close all;
clc;

% 加载图像
Img = imread('Image-Process-Project/testPics/StephenCurry_noisy_sp.jpg');
Img = double(rgb2gray(Img));
figure(1); imshow(uint8(Img), 'InitialMagnification', 'fit'); title('Noisy Image');

% 加载参考图像（无噪声版本）
referenceImg = imread('Image-Process-Project/testPics/StephenCurry.jpeg');
referenceImg = double(rgb2gray(referenceImg));
figure(2); imshow(uint8(referenceImg), 'InitialMagnification', 'fit'); title('Original Image');

[nrow, ncol] = size(Img);

% 准备变量
N = max(nrow, ncol);
alpha = zeros(1, N); 
beta = zeros(1, N); 
gama = zeros(1, N);
u1 = zeros([nrow, ncol]);
u2 = zeros([nrow, ncol]);
timestep = 3;
nn = 3;

% 初始化存储
energies = zeros(1, 5);
psnrs = zeros(1, 5);
ssims = zeros(1, 5);

% 迭代去噪
for n = 1:40
    I_temp = gauss(Img, 3, 1); % 假设gauss是定义好的高斯滤波函数
    Ix = 0.5*(I_temp(:,[2:ncol,ncol])-I_temp(:,[1,1:ncol-1]));
    Iy = 0.5*(I_temp([2:nrow,nrow],:)-I_temp([1,1:nrow-1],:));
    grad = Ix.^2 + Iy.^2;
    g = 1./(1 + grad / 100);

    % 解行
    for i = 1:nrow
        beta(1) = -0.5 * timestep * (g(i, 2) + g(i, 1));
        alpha(1) = 1 - beta(1);
        for j = 2:ncol-1
            beta(j) = -0.5 * timestep * (g(i, j + 1) + g(i, j));
            gama(j) = -0.5 * timestep * (g(i, j - 1) + g(i, j));
            alpha(j) = 1 - beta(j) - gama(j);
        end
        gama(ncol) = -0.5 * timestep * (g(i, ncol) + g(i, ncol - 1));
        alpha(ncol) = 1 - gama(ncol);
        u1(i, :) = Thomas(ncol, alpha, beta, gama, Img(i, :)); % 假设Thomas算法已定义
    end
    
    % 解列
    for j = 1:ncol
        beta(1) = -0.5 * timestep * (g(2, j) + g(1, j));
        alpha(1) = 1 - beta(1);
        for i = 2:nrow-1
            beta(i) = -0.5 * timestep * (g(i + 1, j) + g(i, j));
            gama(i) = -0.5 * timestep * (g(i - 1, j) + g(i, j));
            alpha(i) = 1 - beta(i) - gama(i);
        end
        gama(nrow) = -0.5 * timestep * (g(nrow, j) + g(nrow - 1, j));
        alpha(nrow) = 1 - gama(nrow);
        u2(:, j) = Thomas(nrow, alpha, beta, gama, Img(:, j));
    end
    
    Img = 0.5 * (u1 + u2);
    nn = nn + 1;
    figure(nn); imshow(uint8(Img), 'InitialMagnification', 'fit'); title(sprintf('Iteration %d', n));

    % 计算能量
    energies(n) = calculateEnergy(Img, g);
    % 计算PSNR和SSIM
    psnrs(n) = psnr(uint8(Img), uint8(referenceImg));
    ssims(n) = ssim(uint8(Img), uint8(referenceImg));
end

% 绘制能量演化
figure(nn+1);
plot(1:40, energies, '-o');
xlabel('Iteration Number');
ylabel('Energy');
title('Energy Evolution During Denoising');
grid on;

% 绘制PSNR和SSIM演化
figure(nn+2);
subplot(1, 2, 1);
plot(1:40, psnrs, '-o');
xlabel('Iteration Number');
ylabel('PSNR (dB)');
title('PSNR Evolution During Denoising');
grid on;

subplot(1, 2, 2);
    plot(1:40, ssims, '-o');
xlabel('Iteration Number');
ylabel('SSIM');
title('SSIM Evolution During Denoising');
grid on;
