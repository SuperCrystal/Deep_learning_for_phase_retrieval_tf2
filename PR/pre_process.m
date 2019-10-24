clc,clear
phase_pattern=imread('E:\00_PhaseRetrieval\PR\letters\R2.jpg');
phase_pattern=double(phase_pattern(:,:,1));
maxp=max(max(phase_pattern))
phase_pattern=phase_pattern/maxp;

dim = 10;                 % ��Ļ�߳�1cm
EM =227;                  % ͼƬ�ߴ� 256*256
index=1.5;
f=500;

lambda = 632.8e-6;         % ��λ��mm
k = 2*pi/lambda;
am=400e-6;
z = 500;                   % ������� 10cm

[x,y] =meshgrid(linspace(-dim/2,dim/2,EM));
R = sqrt(x.^2+y.^2);
% R = gpuArray(R);
R(R>dim/2) =0;            % ������Բ��������


wavefront = exp(1j.*k*(index-1)*am*(phase_pattern));
Lens = exp(-1j.*k*R.^2/(2*f)).*cyl(x,y,dim/2);

wave=wavefront.*Lens;
fraun=two_step_prop_ASM(wave,lambda,dim/EM,0.04*dim/EM,z);
% fraun=fftshift(fft2(wavefront));

imwrite(abs(fraun.^2)/20000,'E:\00_PhaseRetrieval\PR\letters\diffractionR2.jpg');

figure(1),
imagesc(abs(fraun.^2)/20000)
colorbar