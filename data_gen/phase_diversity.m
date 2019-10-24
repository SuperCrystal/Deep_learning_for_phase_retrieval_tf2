clc,clear;
image_size=2048;
image_width=10;

n=1.5;
z=150;   
f=150;
lamda=632.8e-6;    %��λ��mm
k=2*pi/lamda;
Amp =1  *lamda;
scale_factor = 0.1;    % ���������е���������
exposure_threshold = 4e4;    % �ع���ֵ����ֵԽС��ʵ���Ӧ���ع�ʱ��Խ������԰�ϸ�ڸ�ͻ����
% period = 0.5;
defocus = 2;

[x,y] =meshgrid(linspace(-image_width/2,image_width/2,image_size));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1=x;
y1=y;
r1=sqrt(x.^2+y.^2);

x=x/(image_width/2);
y=y/(image_width/2);
[theta,r]= cart2pol(x,y);
r(r>=1)=0;

max_zer = 9;  % �����������ε����������

zer = zeros(image_size,image_size,max_zer);


for num = 1:max_zer
    zer(1:image_size,1:image_size,num) = zernike (num, r, theta);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labels=[];
% fp=fopen('E:\00_PhaseRetrieval\TRobj_detec\0722_true.txt','w');                       %'A.txt'Ϊ�ļ�����'w'Ϊ�򿪷�ʽ���ڴ򿪵��ļ�ĩ��������ݣ����ļ��������򴴽���
% phase_true = zeros(image_size,image_size,100);
% PSF_t = zeros(image_size,image_size,100);
% diffraction_true = zeros(image_size,image_size,100);
for count=1:1
%         c_original=[rand(1,10)-0.5 (rand(1,max_zer-10)-0.5)/5];         % �����ϵ��ȡֵ��Χ[-0.5,0.5]
        c_original = [0 rand(1,max_zer-1)-0.5];
        if max_zer>10
            c_original = [c_original(1:10) c_original(11:max_zer)/5];
        else
            c_original=c_original;
        end
        c=c_original;
        s = 0;
        for num  = 1:max_zer
            s = s + c(num) * zer(1:image_size,1:image_size,num);
        end
%             s=s/max(max(abs(s)));
%         max_s = max(max(abs(s)));
%         if max_s > pi/2
%             s = s/max_s*pi/2;
%             c = c/max_s*pi/2;
%         end
        s = s*Amp;                  % ��������ε�PVֵΪ�����amp����(index-1)=0.5
%         s = 0;      
        s_init = s.*cyl(x1,y1,image_width/2);
%         figure(1),
%         ind=find(cyl(x1,y1,image_width/2)==0);
%         mask = cyl(x1,y1,image_width/2);
%         mask(ind)=NaN;
%         mesh(x1,y1,s_init*1000.*mask);
%         colorbar;
%         colormap('jet')
%         caxis([-0.35*lamda*2*1000,0.35*lamda*2*1000])
%         axis square
%         shading interp
%         xlabel('x/mm')
%         ylabel('y/mm')
%         grid off
%         axis off
%         set(get(colorbar,'title'),'string','��m')
        
        u0=exp(-1i*k*r1.^2/(2*f)).*exp(1i*k*s_init*(n-1)).*cyl(x1,y1,image_width/2);
%         phase_true(:,:,count)=s_init;
        uz=two_step_prop_ASM(u0,lamda,image_width/(image_size-1),scale_factor*image_width/(image_size-1),z);
%         uz = ASMDiff(u0,z,lamda,image_width);
        uz_d = two_step_prop_ASM(u0,lamda,image_width/(image_size-1),scale_factor*image_width/(image_size-1),z+defocus);
%         uz_d = ASMDiff(u0,z+defocus,lamda,image_width);
        Iz = abs(uz.*conj(uz));
        Iz_d = abs(uz_d.*conj(uz_d));
%         max(max(Iz))
%         figure(1),
%         plot(x1(256,:),Iz(256,:))
        feature = ift2(ft2(Iz,image_width/(image_size-1))./ft2(Iz_d,image_width/(image_size-1)),image_width/(image_size-1));
        feature=feature/max(max(feature))*255 ;
        Iz(Iz>exposure_threshold) = exposure_threshold; % ģ�����
        Iz_d(Iz_d>exposure_threshold) = exposure_threshold;
        
        max_I = max(max(Iz));
        Iz=Iz/max_I*255 ;   % ��һ������ 
        Iz_d=Iz_d/max_I*255 ;
%       current_label=[c(1:10) c(11:max_zer)*5];                                                     % �������ε�ʱ���ú�10�����ƣ���¼ϵ����ʱ���һ����ͬһscale��������ѵ��������
      
      %%%
      current_name=['image' num2str(count,'%06d') '.bmp'];
%       current_name_and_label=num2str(current_label,' %.8f');
%       fprintf(fp,'%s \r\n',current_name_and_label);                         % fpΪ�ļ������ָ��Ҫд�����ݵ��ļ���ע�⣺%d���пո�
%       fprintf(fperiod,'%s \r\n',num2str(period, '%.8f'));                 % ��¼������period 
      %����ͼ��
      save_name_phase = ['E:\00_PhaseRetrieval\PhENN\data_gen\test\phase\' current_name];
      save_name_intensity = ['E:\00_PhaseRetrieval\PhENN\data_gen\test\intensity\' current_name];
      save_name_intensity_de = ['E:\00_PhaseRetrieval\PhENN\data_gen\test\intensity\de_' current_name];
      imwrite(uint8(Iz),save_name_intensity,'bmp');        % double��ͼ��imwriteʱ��Ϊ0-1��
      imwrite(uint8(Iz_d),save_name_intensity_de,'bmp');
%       iii = ifft2(fft2(Iz)./fft2(Iz_d));
      imwrite(uint8(feature), 'E:\00_PhaseRetrieval\PhENN\data_gen\test\intensity\feature.bmp', 'bmp')
      s_save = (s_init/lamda+2)/4*255.*cyl(x1,y1,image_width/2);         % s_init��һ������Ÿ�׼ȷ��
      imwrite(uint8(s_save),save_name_phase, 'bmp');
%       imwrite(s_init/max(max(abs(s_init))),['E:\00_PhaseRetrieval\TRobj_detec\surface\' current_name]);
      
%       diffraction_true(:,:,count) = Iz;
end
% max(max(phase_true/lamda))
% min(min(phase_true/lamda))

% fclose(fp);%�ر��ļ���      
% save('E:\00_PhaseRetrieval\TRobj_detec\0722_phase_true','phase_true');%����test�ļ����ɣ���train�ļ�����
% save('E:\00_PhaseRetrieval\TRobj_detec\0722_PSF_t','PSF_t');
% save('E:\00_PhaseRetrieval\GS�㷨\diffraction_true','diffraction_true');
