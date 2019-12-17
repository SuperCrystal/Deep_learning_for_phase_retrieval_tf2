% ���ع�Ĵ�ͳHIO�㷨�����100��test sample�����ʣ��rms�������Ա�
clc;
clear all;
% close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_width = 10;
image_size =512;
n = 1.5;
lambda = 632.8e-6;
Amp = 0.2  *lambda;
k = 2*pi/lambda;
f = 500;
z = 500;
n_iter = 200;
scale_factor = 0.1;    % ���������е���������
gt_label_file = 'test';
predict_file = 'single';
max_zer = 20;  % �����������ε����������
test_case_num = 100;
c_predicts = load(['E:\00_PhaseRetrieval\PhENN\result\' predict_file '\results.txt']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
for count=1:test_case_num
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    current_mat_name=['image' num2str(count,'%06d')];
    c_original = load(['E:\00_PhaseRetrieval\PhENN\dataset\' gt_label_file '\phase\' current_mat_name '.txt']);
    c_predict = c_predicts(count,:)-0.5;

    s = zernike_sur(c_original, max_zer, image_size, r1, theta)*Amp;    % ground truth����
    s_p = zernike_sur(c_predict, max_zer, image_size, r1, theta)*Amp;   % predict������
    % s_p=0;
    phase_True = angle(exp(1i*k*s*(n-1)).*cyl(x,y,image_width/2));
    phase_zer = exp(1i*k*s*(n-1));
    phase_len = exp(-1i*k*R.^2/(2*f));
    phase_total = phase_len.*phase_zer.*cyl(x,y,image_width/2);
    phase_pred = phase_len.*exp(1i*k*s_p*(n-1)).*cyl(x,y,image_width/2);
    %%%%%%%%%%% ��ģ %%%%%%%%%%%%%
    mask = zeros(size(phase_total)*2);
    mask= gpuArray(mask);
    EM_mask = size(mask,1);
    mask(EM_mask/2-image_size/2:EM_mask/2+image_size/2-1,EM_mask/2-image_size/2:EM_mask/2+image_size/2-1)=cyl(x,y,image_width/2);
    img_mask = mask;
    img_mask(mask==1) = phase_total(cyl(x,y,image_width/2)==1);
    phase_total = img_mask;
%     figure(1)
%     imagesc(angle(phase_zer))
%     colorbar;
    %%%%%%%%% �������۹�ǿ���ڵ����ָ��������ʾ��õĹ�ǿ�Ƿǳ�����ģ������أ������������뽹��
    u0 = phase_total;
    uz=two_step_prop_ASM(u0,lambda,delta1,delta2,z);
    T = sqrt(abs(uz.*conj(uz)));
    % ����
    % ����
    % �뽹
    %%%%%%%%%%%% ѡһ������ʼֵ
    % ͸������ʼ��ֵ
    temp = mask;
    temp(mask==1) = phase_len(cyl(x,y,image_width/2)==1);
    phase_len = temp;
    phase_init = phase_len;
%     figure(4);
%     imagesc(angle(phase_len))
    % phase_predҲ��Ҫ��mask
    temp2 = mask;
    temp2(mask==1) = phase_pred(cyl(x,y,image_width/2)==1);
    phase_pred = temp2;
%     figure(5);
%     imagesc(angle(phase_pred./phase_len))
    %%%%%%%%%%%%%%%%%%%%% HIO�㷨���� %%%%%%%%%%%%%%%%%%%%%
    [E_plane_output, err] = hio_func(T,phase_pred,image_width,n_iter,mask,phase_True,phase_len,lambda,delta1,delta2,z);
%     figure(2);
%     imagesc(angle(E_plane_output./phase_len))
%     colorbar
%     figure(3),
%     plot(err)
%     set(gca,'YLim',[0 0.2])
    
    rmse_final(count) = err(n_iter); 
end
average_rmse = sum(rmse_final)/test_case_num    % ��ֵ���
% rmse = 

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
% ���100������΢������sample

% ���100���������΢�뽹��sample