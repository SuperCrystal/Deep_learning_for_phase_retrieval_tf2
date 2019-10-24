clc
clear all
close all
tic
dim = 10;                 % 屏幕边长4cm
EM =227;                  % 图片尺寸 256*256

index = 1.5;
Amp = 400e-6;

lambda = 632.8e-6;         % 单位：mm
k = 2*pi/lambda;
f = 500;
z = 500;                   % 衍射距离 10cm


n_iter = 1000;


[x,y] =meshgrid(linspace(-dim/2,dim/2,EM));
R = sqrt(x.^2+y.^2);
% R = gpuArray(R);
R(R>dim/2) =0;            % 限制在圆形区域内

Lens = exp(-1j.*k*R.^2/(2*f)).*cyl(x,y,dim/2);

x1=x;
y1=y;



T= sqrt(50000*double(imread('diffractionR1.jpg')));
T=T(:,:,1);

nr_of_phaselevels = 256;

x=x/(dim/2);
y=y/(dim/2);
z1=x;
z2=y;
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

c(1)=(dim/EM)/(2*f)*dim/2/Amp;
c(2)=(dim/EM)/(2*f)*dim/2/Amp;  % 论文上应该是zernikes为相位时系数a的求法
c(3:15)=1/10*[-4.869988   -1.5792269   1.0544752  -0.36361602  0.20093885  0.17519355  -0.7942497   0.1411737   2.7515507   0.08124541 -1.1903822  -0.39976826  -6.4414864];
s=c(3)*z3+c(4)*z4+c(5)*z5+c(6)*z6+c(7)*z7+c(8)*z8+c(9)*z9+c(10)*z10+c(11)*z11+c(12)*z12+c(13)*z13+c(14)*z14+c(15)*z15;
s=s.*cyl(x1,y1,dim/2);
% s=c(3)*z3+c(4)*z4+c(5)*z5+c(6)*z6+c(7)*z7+c(8)*z8+c(9)*z9+c(10)*z10+c(11)*z11+c(12)*z12+c(13)*z13+c(14)*z14+c(15)*z15;
predicted_phase=exp(1j.*k*(index-1)*Amp*s);
figure(3),mesh(x1,-y1,angle(predicted_phase));

Phase_pattern=double(imread('R1.jpg'));
Phase_pattern=Phase_pattern(:,:,1);
Phase_pattern=Phase_pattern/max(max(Phase_pattern));
wavefront = exp(1j.*k*(index-1)*Amp*(Phase_pattern));
Phase_true=angle(wavefront);
figure(1)
% imagesc(Phase_true)
mesh(x1,-y1,Phase_true)
colorbar
colormap winter
caxis([-pi,pi])

% E_plane_1 = Lens.*predicted_phase;
% E_plane_1 = Lens;
E_plane_1 = predicted_phase;


for i = 1:n_iter
    E_plane_2 = two_step_prop_ASM(E_plane_1,lambda,dim/EM,0.04*dim/EM,z);%将估计值衍射到光强采集面
    E_plane_2_new = T.*exp(1j.*angle(E_plane_2));%用实际采集到光强图替换计算值
    E_plane_1_new = two_step_prop_ASM(E_plane_2_new,lambda,0.04*dim/EM,dim/EM,-z);%逆衍射回空间域
    phase = angle(E_plane_1_new);%获取相位
    E_plane_1 = exp(1j.*phase).*cyl(x1,y1,dim/2);%空间域限制
%     E_plane_1 = exp(1j.*phase);
%     E_plane_1(R>dim/2)=0;
    E_plane_1(E_plane_1==0)=eps;
    phase_final = (angle(E_plane_1./Lens));
    err(i)=sqrt(sum(sum(abs((Phase_true-phase_final).^2))))./EM./EM;%RMS
end

figure(2)
phase_final = (angle(E_plane_1./Lens));
mesh(x1,-y1,angle(E_plane_1./Lens))
colorbar
colormap winter
caxis([-pi,pi])
% figure(3)
% plot(phase_final(:,round(EM/2)))
figure(4)
plot(err)