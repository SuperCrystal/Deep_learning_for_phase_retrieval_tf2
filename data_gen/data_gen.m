clc,clear;
image_size=512;
image_width=10;

n=1.5;
z=500;   
f=500;
lamda=632.8e-6;    %��λ��mm
k=2*pi/lamda;
Amp = 0.2  *lamda;
scale_factor = 0.1;    % ���������е���������
exp_thresh = [0.5e4 1.5e4 2.5e4];    % �ع���ֵ����ֵԽС��ʵ���Ӧ���ع�ʱ��Խ������԰�ϸ�ڸ�ͻ����exp_thresh_1 = 0.5e4;

% period = 0.5;
[x,y] =meshgrid(linspace(-image_width/2,image_width/2,image_size));
target_file = 'test';       % 'train' 'validate' 'test'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1=x;
y1=y;
r1=sqrt(x.^2+y.^2);

x=x/(image_width/2);
y=y/(image_width/2);
[theta,r]= cart2pol(x,y);
r(r>=1)=0;

max_zer = 20;  % �����������ε����������

zer = zeros(image_size,image_size,max_zer);


for num = 1:max_zer
    zer(1:image_size,1:image_size,num) = zernike (num, r, theta);
end

for count=1:100
        c_original = rand(1,max_zer)-0.5;
        c = c_original;
        s = 0;
        for num  = 1:max_zer
            s = s + c(num) * zer(1:image_size,1:image_size,num);
        end
        s = s*Amp;                  % ��������ε�PVֵΪ�����amp����(index-1)=0.5
%         s = 0;      
        s_init = s.*cyl(x1,y1,image_width/2);
        u0=exp(-1i*k*r1.^2/(2*f)).*exp(1i*k*s_init*(n-1)).*cyl(x1,y1,image_width/2);
%         phase_true(:,:,count)=s_init;
        uz=two_step_prop_ASM(u0,lamda,image_width/(image_size-1),scale_factor*image_width/(image_size-1),z);
        Iz=abs(uz.*conj(uz));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ����ģ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Iz_exp1 = Iz;
%         Iz_exp2 = Iz;
%         Iz_exp3 = Iz;
%         Iz_exp1(Iz_exp1>exp_thresh(1)) = exp_thresh(1); % ģ������͹���
%         Iz_exp2(Iz_exp2>exp_thresh(2)) = exp_thresh(2);
%         Iz_exp3(Iz_exp3>exp_thresh(3)) = exp_thresh(3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ����ģ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Iz = imnoise(Iz,'gaussian', 0, 0.01);

      current_label = c;
      current_name=['image' num2str(count,'%06d') '.bmp'];
      current_mat_name=['image' num2str(count,'%06d')];
      %����ͼ��
      save_name_intensity_mat = ['E:\00_PhaseRetrieval\PhENN\dataset\' target_file '\intensity\' current_mat_name];
      save(save_name_intensity_mat, 'Iz');
      fp = fopen(['E:\00_PhaseRetrieval\PhENN\dataset\' target_file '\phase\' current_mat_name '.txt'], 'w');
      fprintf(fp, '%s \r\n', num2str(current_label,' %.8f'));
      fclose(fp);
      
end
% max(max(max(phase_true/lamda)))
% min(min(min(phase_true/lamda)))

