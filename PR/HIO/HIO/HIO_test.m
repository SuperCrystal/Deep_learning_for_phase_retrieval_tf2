clc;
clear all;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 参数 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_width = 10;
image_size =512;
n = 1.5;
lambda = 632.8e-6;
Amp = 0.2  *lambda;
k = 2*pi/lambda;
f = 500;
z = 500;
n_iter = 1000;
scale_factor = 0.1;    % 两步角谱中的缩放因子
delta1 = image_width/(image_size-1);
delta2 = scale_factor*image_width/(image_size-1);
[x,y] =meshgrid(linspace(-image_width/2,image_width/2,image_size));
R = sqrt(x.^2+y.^2);
R= gpuArray(R);
R(R>image_width/2) =0;
x1=x/(image_width/2);
y1=y/(image_width/2);
[theta,r1]= cart2pol(x1,y1);
r1(r1>=1)=0;
max_zer = 20;  % 用来生成面形的泽尼克项数
test_case_num = 100;
for count=1:test_case_num
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 面形生成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % c_original = load('E:\00_PhaseRetrieval\PhENN\dataset\train\phase\image000001.txt');
%     count = 2;
    current_mat_name=['image' num2str(count,'%06d')];
    c_original = load(['E:\00_PhaseRetrieval\PhENN\dataset\train\phase\' current_mat_name '.txt']);
    c_predict = load(['E:\00_PhaseRetrieval\PhENN\dataset\predict\phase\' current_mat_name '.txt']);
%     c_predict = [0.00647449 -0.30987000  0.19227306  0.39816113 -0.17486398 -0.06265780 -0.48511722  0.12137747  0.307168198 -0.20309292 -0.43538997 -0.37204780  0.10783658  0.29242091  0.10101045 -0.03079385 -0.10563767  0.13146666  0.0565202690  0.10372667 ];
    c_predict = [0.1 -0.4 0.1922 0.39816113 -0.17486398 zeros(1,15)];%1 泽尼克项偏差小于0.1似乎HIO恢复效果就不错了
    % c_predict = [0.1 -0.2 zeros(1,18)];%2
%     c_predict = rand(1,20)-0.5;

    s = zernike_sur(c_original, max_zer, image_size, r1, theta)*Amp;    % ground truth面形
    s_p = zernike_sur(c_predict, max_zer, image_size, r1, theta)*Amp;   % predict的面形
    % s_p=0;
    phase_True = angle(exp(1i*k*s*(n-1)).*cyl(x,y,image_width/2));
    phase_zer = exp(1i*k*s*(n-1));
    phase_len = exp(-1i*k*R.^2/(2*f));
    phase_total = phase_len.*phase_zer.*cyl(x,y,image_width/2);
    phase_pred = phase_len.*exp(1i*k*s_p*(n-1)).*cyl(x,y,image_width/2);
    %%%%%%%%%%% 掩模 %%%%%%%%%%%%%
    mask = zeros(size(phase_total)*2);
    mask= gpuArray(mask);
    EM_mask = size(mask,1);
    mask(EM_mask/2-image_size/2:EM_mask/2+image_size/2-1,EM_mask/2-image_size/2:EM_mask/2+image_size/2-1)=cyl(x,y,image_width/2);
    img_mask = mask;
    img_mask(mask==1) = phase_total(cyl(x,y,image_width/2)==1);
    phase_total = img_mask;
    figure(1)
    imagesc(angle(phase_zer))
    colorbar;
    %%%%%%%%% 计算理论光强用于迭代恢复，这里表示获得的光强是非常理想的，不过曝，无噪声，无离焦。
    u0 = phase_total;
    uz=two_step_prop_ASM(u0,lambda,delta1,delta2,z);
    T = sqrt(abs(uz.*conj(uz)));
    temp = mask;
    temp(mask==1) = phase_len(cyl(x,y,image_width/2)==1);
    phase_len = temp;
    % 过曝
    % 噪声
    % 离焦
    %%%%%%%%%%%% 选一个做初始值
    % 透镜做初始化值
    phase_init = phase_len;
    % phase_pred也需要加mask
    temp2 = mask;
    temp2(mask==1) = phase_pred(cyl(x,y,image_width/2)==1);
    phase_pred = temp2;
    % figure(2)
    % imagesc(T)
    % colorbar;
    %%%%%%%%%%%%%%%%%%%%% HIO算法迭代 %%%%%%%%%%%%%%%%%%%%%
    [E_plane_output, err] = hio_func(T,phase_pred,image_width,n_iter,mask,phase_True,phase_len,lambda,delta1,delta2,z);
    figure(2);
    imagesc(angle(E_plane_output./phase_len))
    colorbar
    % figure(3)
    % T_phase = angle(E_plane_output);
    % plot(T_phase(:,image_size))
    % figure(3),
    % plot(err)
    rmse_final(count) = err(n_iter); 
end
average_rmse = rmse_final/test_case_num;    % 均值误差
rmse = 

function s = zernike_sur(c_original, max_zer, image_size, r1, theta)
zer = zeros(image_size,image_size,max_zer);
for num = 1:max_zer
    zer(1:image_size,1:image_size,num) = zernike (num, r1, theta);
end
s = 0;
for num  = 1:max_zer
    s = s + c_original(num) * zer(1:image_size,1:image_size,num);
end
end