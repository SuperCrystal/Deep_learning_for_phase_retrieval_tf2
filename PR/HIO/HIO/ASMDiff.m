function Uz1=ASMDiff(T1,z1,lambda,dim)
    EM=size(T1,1);
    k=2*pi/lambda;
    Lx=dim;
    Ly=dim;
    Mx=EM;
    My=EM;
%% From Homework2 of Feihong Yu's Class
    dx=Lx/Mx;
    dy=Ly/My;%sample interval
%     X=-Lx/2:dx:Lx/2-dx;
%     Y=-Ly/2:dy:Ly/2-dy;%coordinate vector of 2D
%     [x,y]=meshgrid(X,Y);
    Fx=-1/(2*dx):1/Lx:1/(2*dx)-(1/Lx);%����ռ�Ƶ�ʣ���λ�����µĿռ�������
    Fy=-1/(2*dy):1/Ly:1/(2*dy)-(1/Ly);%����ռ�Ƶ��
    [fx,fy]=meshgrid(Fx,Fy);
   
%% ��������
    A=exp(1j*k*z1*sqrt(1-lambda^2*(fx.^2+fy.^2)));
    A(sqrt(fx.^2+fy.^2)>=1/lambda)=0;
    Tf1=fftshift(fft2(T1));%͸�亯��Ƶ��
    Uz1=(ifft2(fftshift(A.*Tf1)));
end
