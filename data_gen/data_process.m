% ���������train��Iz��mat���ݣ����䴦��ɹ������ݲ�����
clc,clear;

source_file = 'test';
target_file = 'noise_test_for_cal';       % 'train' 'validate' 'test'
exp_thresh = [0.1e4 0.5e4 3e4];    % �ع���ֵ����ֵԽС��ʵ���Ӧ���ع�ʱ��Խ������԰�ϸ�ڸ�ͻ����exp_thresh_1 = 0.5e4;
                                   % 16λ���0-6.5e4
total_img = 100;
for count=1:total_img
    current_mat_name=['image' num2str(count,'%06d')];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% label���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    save_phase_name = ['image' num2str(count+total_img,'%06d')];
    current_label = load(['E:\00_PhaseRetrieval\PhENN\dataset\' source_file '\phase\' current_mat_name '.txt']);
    fp = fopen(['E:\00_PhaseRetrieval\PhENN\dataset\' target_file '\phase\' save_phase_name '.txt'], 'w');
    fprintf(fp, '%s \r\n', num2str(current_label,' %.8f'));
    fclose(fp);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ͼƬ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    save_name_intensity_mat = ['E:\00_PhaseRetrieval\PhENN\dataset\' source_file '\intensity\' current_mat_name '.mat'];
    load(save_name_intensity_mat);
    %%%% ����ģ�� %%%%
    max_I = max(max(Iz));
%     Iz(Iz>max_I) = max_I;  % ����һ����������̬��Χ
    Iz_exp1 = Iz;
    Iz_exp2 = Iz;
    Iz_exp3 = Iz;
    % ����(��ֵ��������ĸ�˹������
    noise_scale = rand(1)*0.2+0.1;
    mean = 0;
    sigma = rand(1)*0.02+0.01;
    temp1 = Iz_exp1;
    temp2 = Iz_exp2;
    temp3 = Iz_exp3;
%     Iz_exp1 = Iz_exp1/max_I;
    Iz_exp1 = Iz_exp1 + noise_scale*imnoise(Iz_exp1/max_I,'gaussian',mean,sigma);
    Iz_exp2 = Iz_exp2 + noise_scale*imnoise(Iz_exp2/max_I,'gaussian',mean,sigma);
    Iz_exp3 = Iz_exp3 + noise_scale*imnoise(Iz_exp3/max_I,'gaussian',mean,sigma);
    iii1 = Iz_exp1-temp1;
    iii2 = Iz_exp2-temp2;
    iii3 = Iz_exp3-temp3;
%     n1 = sum(sum(abs((Iz_exp1-temp1))./temp1))/512^2;
    mse1 = sum(sum(iii1.^2))/(512^2)
    mse2 = sum(sum(iii2.^2))/(512^2)
    mse3 = sum(sum(iii3.^2))/(512^2)
    psnr1 = 10*log10(1/mse1)
    psnr2 = 10*log10(1/mse2)
    psnr3 = 10*log10(1/mse3)
%     n2 = sum(sum(abs((Iz_exp3-temp3))./temp3))/512^2;
    % ����
    Iz_exp1(Iz_exp1>exp_thresh(1)) = exp_thresh(1); % ģ������͹���
    Iz_exp2(Iz_exp2>exp_thresh(2)) = exp_thresh(2);
    Iz_exp3(Iz_exp3>exp_thresh(3)) = exp_thresh(3);
    % ֱ����������й�һ����
    Iz_exp1 = Iz_exp1/exp_thresh(1);
    Iz_exp2 = Iz_exp2/exp_thresh(2);
    Iz_exp3 = Iz_exp3/exp_thresh(3);
        % ����
%     noise_scale = 1;
%     mean = 0;
%     sigma = 0.02;
%     Iz_exp1 = noise_scale*imnoise(Iz_exp1,'gaussian',mean,sigma);
%     Iz_exp2 = noise_scale*imnoise(Iz_exp2,'gaussian',mean,sigma);
%     Iz_exp3 = noise_scale*imnoise(Iz_exp3,'gaussian',mean,sigma);
    % ���ӻ�����ͼ��
    figure(1),
    colormap('hot')
    imagesc(Iz_exp1)
    figure(2),
    colormap('hot')
    imagesc(Iz_exp2)
    figure(3),
    colormap('hot')
    imagesc(Iz_exp3)

    % �������ͼ��
    save_mat_name=['image' num2str(count+total_img,'%06d')];    % noiseͼƬ�����ִ�10001��ʼ������������ѵ����
    save_1 = ['E:\00_PhaseRetrieval\PhENN\dataset\' target_file '\intensity_1\' save_mat_name];
    save_2 = ['E:\00_PhaseRetrieval\PhENN\dataset\' target_file '\intensity_2\' save_mat_name];
    save_3 = ['E:\00_PhaseRetrieval\PhENN\dataset\' target_file '\intensity_3\' save_mat_name];
    save(save_1, 'Iz_exp1');
    save(save_2, 'Iz_exp2');
    save(save_3, 'Iz_exp3');
end


