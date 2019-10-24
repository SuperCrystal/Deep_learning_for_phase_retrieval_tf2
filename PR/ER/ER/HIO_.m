clc
clear all
tic
dim = 10;
EM =1024;
period = 1;
index = 1.5;
Amp = 200E-6;
lambda = 632.8e-6;         %单位：mm
k = 2*pi/lambda;
f = 80;
z = 80;


n_iter = 20;


[x,y] =meshgrid(linspace(-dim/2,dim/2,EM));
R = sqrt(x.^2+y.^2);
% R = gpuArray(R);
R(R>dim/2) =0;

grating = exp(1j.*k*(index-1)*Amp*cos(2*pi*R./period)).*cyl(x,y,dim/2);

Lens = exp(-1j.*k*R.^2/(2*f)).*cyl(x,y,dim/2);
Phase_grating = angle(grating);

figure(1)
imagesc(angle(grating))
colorbar
caxis([-pi,pi])

U = grating.*Lens;
T = abs(two_step_prop_ASM(U,lambda,dim/(EM-1),0.05*dim/(EM-1),z)).^2;%获取实际光强图 0.00375
T = sqrt(T);
figure(5)
imagesc(T)
figure(6)
plot(linspace(-dim/2,dim/2,EM),T(EM/2,:))

nr_of_phaselevels = 256;
E_plane_1 = Lens;
for i = 1:n_iter
    E_plane_2 = two_step_prop_ASM(E_plane_1,lambda,dim/(EM-1),0.05*dim/(EM-1),z);%将估计值衍射到光强采集面
    E_plane_2_new = T.*exp(1j.*angle(E_plane_2));%用实际采集到光强图替换计算值
    E_plane_1_new = two_step_prop_ASM(E_plane_2_new,lambda,0.05*dim/(EM-1),dim/(EM-1),-z);%逆衍射回空间域
%     phase = angle(E_plane_1_new);%获取相位
%     E_plane_1 = exp(1j.*phase).*cyl(x,y,dim/2);%空间域限制
    beta = 0.9;
    E_plane_1 = E_plane_1-beta*E_plane_1_new;
    E_plane_1(E_plane_1==0)=eps;
    phase_final = (angle(E_plane_1_new./Lens));
    err(i)=sum(sum(sqrt(abs(Phase_grating.^2-phase_final.^2))))./EM./EM;%RMS
end

figure(2)
phase_final = (angle(E_plane_1./Lens));
imagesc(angle(E_plane_1./Lens))
colorbar
caxis([-pi,pi])
figure(3)
plot(phase_final(:,EM/2))
figure(4)
plot(err)