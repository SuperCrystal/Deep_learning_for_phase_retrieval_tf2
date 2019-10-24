clc,clear;
image_size=227;
image_width=10;


n=1.5;
z=500;   
f=500;
lamda=632.8e-6;    %单位：mm
k=2*pi/lamda;

[x,y] =meshgrid(linspace(-image_width/2,image_width/2,image_size));

r=sqrt(x.^2+y.^2);
r(r>=image_width/2)=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x1=x;
y1=y;

x=x/(image_width/2);
y=y/(image_width/2);
z3=-1+2*(x.^2+y.^2);
z4=x.^2-y.^2;
z5=2*x.*y;
z6=-2*x+3*x.*(x.^2+y.^2);
z7=-2*y+3*y.*(x.^2+y.^2);
z8=1-6*(x.^2+y.^2)+6*(x.^2+y.^2).^2;
z9=x.^3-3*x.*(y.^2);
z10=-y.^3+3*y.*(x.^2);
z11=-3*x.^2+3*y.^2+4*(x.^2).*(x.^2+y.^2)-4*(y.^2).*(x.^2+y.^2);
z12=-6*x.*y+8*x.*y.*(x.^2+y.^2);
z13=3*x-12*x.*(x.^2+y.^2)+10*x.*(x.^2+y.^2).^2;
z14=3*y-12*y.*(x.^2+y.^2)+10*y.*(x.^2+y.^2).^2;
z15=-1+12*(x.^2+y.^2)-30*(x.^2+y.^2).^2+20*(x.^2+y.^2).^3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labels=[];
fp=fopen('E:\00_PhaseRetrieval\PhENN\train\train.txt','w');%'A.txt'为文件名；'a'为打开方式：在打开的文件末端添加数据，若文件不存在则创建。
rms=0.5*lamda;   %rms为2.3λ
% temp=zeros(14580,227,227,3);
for count=1:1:300
            c_original=2*rand(1,15)-1;
            c=c_original;
%             c=c_original/1000;
            s=c(3)*z3+c(4)*z4+c(5)*z5+c(6)*z6+c(7)*z7+c(8)*z8+c(9)*z9+c(10)*z10+c(11)*z11+c(12)*z12+c(13)*z13+c(14)*z14+c(15)*z15;   % 为使相位在-pi-pi，s应在-1到1之间
%             s=s/max(max(abs(s)));
            c=c/3;
            s=(c(3)*z3+c(4)*z4+c(5)*z5+c(6)*z6+c(7)*z7+c(8)*z8+c(9)*z9+c(10)*z10+c(11)*z11+c(12)*z12+c(13)*z13+c(14)*z14+c(15)*z15).*cyl(x1,y1,image_width/2);
%             max(max(abs(s)))
%             max(max(c))
%             crms=sqrt(sum(sum(s.^2))/image_size^2);

            u0=exp(-1i*k*r.^2/(2*f)).*exp(1i*k*400e-6*s*(n-1)).*cyl(x1,y1,image_width/2);
%             figure(1)
%             imagesc(angle(u0))
%             colorbar
%             figure(2)
%             imagesc(angle(exp(1i*k*400e-6*s*(n-1))))
%             colorbar
%             caxis([-pi,pi])

            uz=two_step_prop_ASM(u0,lamda,image_width/image_size,0.04*image_width/image_size,z);
           
            Iz=abs(uz.*conj(uz));

            Iz=Iz/50000;
% max(max(Iz))
            Irgb=cat(3,Iz,Iz,Iz);
%             figure(1)
%             imagesc(Iz)
%             colorbar
            %每次存下图片名和系数值，最终labels为需要存成txt的矩阵
%             current_label=c_original(3:15); 
            current_label=c(3:15)*10;                          % 系数木有单位
            current_name=['image' num2str(count,'%06d') '.jpg'];
            %current_name_and_label=[current_name ' ' num2str(current_label,' %.8f')];      
            current_name_and_label=num2str(current_label,' %.8f');
            fprintf(fp,'%s \r\n',current_name_and_label);%fp为文件句柄，指定要写入数据的文件。注意：%d后有空格。
            %保存图像
            save_name=['E:\00_PhaseRetrieval\PhENN\train\train\' current_name];
            imwrite(Irgb,save_name);
            temp(count,:,:,:)=Irgb;
end
% average=sum(sum(sum(sum(temp))))/(300*256*256*3)
max(max(max(max(temp))))
min(min(min(min(temp))))
fclose(fp);%关闭文件。            
