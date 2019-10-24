clc,clear;
image_size=227;
image_width=10;

Amp = 200E-6;
n=1.5;
z=500;   
f=500;
lamda=632.8e-6;    %单位：mm
k=2*pi/lamda;

[x,y] =meshgrid(linspace(-image_width/2,image_width/2,image_size));

r=sqrt(x.^2+y.^2);
r(r>=image_width/2)=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labels=[];
fp=fopen('E:\00_PhaseRetrieval\PhENN\train\trainmsf.txt','w');%'A.txt'为文件名；'a'为打开方式：在打开的文件末端添加数据，若文件不存在则创建。

for count=1:1:20
%             max(max(abs(s)))
%             max(max(c))
%             crms=sqrt(sum(sum(s.^2))/image_size^2);
            period = 0.1 + 0.0004 * (count-1);
            grating = exp(1j.*k*(n-1)*Amp*cos(2*pi*r./period)).*cyl(x,y,image_width/2);
            u0=exp(-1j*k*r.^2/(2*f)).*grating;
%             figure(1)
%             imagesc(angle(u0))
%             colorbar
%             figure(2)
%             imagesc(angle(exp(1i*k*400e-6*s*(n-1))))
%             colorbar
%             caxis([-pi,pi])

            uz=two_step_prop_ASM(u0,lamda,image_width/image_size,0.03*image_width/image_size,z);
           
            Iz=abs(uz.*conj(uz));

            Iz=Iz/50000;
% max(max(Iz))
            Irgb=cat(3,Iz,Iz,Iz);
%             figure(1)
%             imagesc(Iz)
%             colorbar
            %每次存下图片名和系数值，最终labels为需要存成txt的矩阵
%             current_label=c_original(3:15); 
            current_label = period;                          % 系数木有单位
            current_name=['image' num2str(count,'%06d') '.jpg'];
            %current_name_and_label=[current_name ' ' num2str(current_label,' %.8f')];      
            current_name_and_label=num2str(current_label,' %.8f');
            fprintf(fp,'%s \r\n',current_name_and_label);%fp为文件句柄，指定要写入数据的文件。注意：%d后有空格。
            %保存图像
            save_name=['E:\00_PhaseRetrieval\PhENN\train\trainmsf\' current_name];
            imwrite(Irgb,save_name);
            temp(count,:,:,:)=Irgb;
end
% average=sum(sum(sum(sum(temp))))/(300*256*256*3)
max(max(max(max(temp))))
min(min(min(min(temp))))
fclose(fp);%关闭文件。            
