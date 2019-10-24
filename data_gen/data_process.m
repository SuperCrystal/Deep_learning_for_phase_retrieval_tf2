% ���������train��Iz��mat���ݣ����䴦��ɹ������ݲ�����
clc,clear;

target_file = 'exp_train';       % 'train' 'validate' 'test'
exp_thresh = [0.1e4 0.5e4 3e4];    % �ع���ֵ����ֵԽС��ʵ���Ӧ���ع�ʱ��Խ������԰�ϸ�ڸ�ͻ����exp_thresh_1 = 0.5e4;

for count=1:10000
    current_mat_name=['image' num2str(count,'%06d')];
    save_name_intensity_mat = ['E:\00_PhaseRetrieval\PhENN\dataset\train\intensity\' current_mat_name '.mat'];
    load(save_name_intensity_mat);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ����ģ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Iz_exp1 = Iz;
    Iz_exp2 = Iz;
    Iz_exp3 = Iz;
    Iz_exp1(Iz_exp1>exp_thresh(1)) = exp_thresh(1); % ģ������͹���
    Iz_exp2(Iz_exp2>exp_thresh(2)) = exp_thresh(2);
    Iz_exp3(Iz_exp3>exp_thresh(3)) = exp_thresh(3);
    % ֱ����������й�һ����
    Iz_exp1 = Iz_exp1/exp_thresh(1);
    Iz_exp2 = Iz_exp2/exp_thresh(2);
    Iz_exp3 = Iz_exp3/exp_thresh(3);
    % ���ӻ�����ͼ��
%     figure(1),
%     imagesc(Iz_exp1)
%     figure(2),
%     imagesc(Iz_exp2)
%     figure(3),
%     imagesc(Iz_exp3)

    % �������ͼ��
    save_1 = ['E:\00_PhaseRetrieval\PhENN\dataset\' target_file '\intensity_1\' current_mat_name];
    save_2 = ['E:\00_PhaseRetrieval\PhENN\dataset\' target_file '\intensity_2\' current_mat_name];
    save_3 = ['E:\00_PhaseRetrieval\PhENN\dataset\' target_file '\intensity_3\' current_mat_name];
    save(save_1, 'Iz_exp1');
    save(save_2, 'Iz_exp2');
    save(save_3, 'Iz_exp3');
end


