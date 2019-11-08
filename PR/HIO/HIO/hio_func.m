function [E_plane_output, error_OSS]= hio_func(T,phase_init,dim,n_iter,mask,phase_T,phase_len,lambda,delta1,delta2,z)

% dim =10;
beta1 = 0.9;
% rand('seed',0);
% phase_init = rand(size(T));
phase_init = gpuArray(phase_init );
%%

phase_init = angle(two_step_prop_ASM(phase_init,lambda,delta1,delta2,z));
E_plane_2 = T.*exp(1j.*phase_init);
E_plane_1 = two_step_prop_ASM(E_plane_2,lambda,delta2,delta1,-z);
EM_mask = size(T,1);
EM = EM_mask/2;
[x,y] =meshgrid(linspace(-dim/2,dim/2,EM));
% phase_T = (phase_T-min(min(phase_T))).*cyl(x,y,dim/2);
phase_T = phase_T.*cyl(x,y,dim/2);
for t = 1:n_iter
    rs = two_step_prop_ASM(E_plane_2,lambda,delta2,delta1,-z);
    
    cond1 =~mask;
    rs(cond1) = E_plane_1(cond1)-beta1.*rs(cond1);
    E_plane_1  = rs;
    E_plane_2_new = two_step_prop_ASM(rs,lambda,delta1,delta2,z);
    
    E_plane_2 = T.*exp(1j.*angle(E_plane_2_new));
    %%
    E_plane_output = two_step_prop_ASM(E_plane_2,lambda,delta2,delta1,-z);
    T_phase = angle(E_plane_output./phase_len);
    output_phase = T_phase(EM_mask/2-EM/2:EM_mask/2+EM/2-1,EM_mask/2-EM/2:EM_mask/2+EM/2-1).*cyl(x,y,dim/2);
    % 这里可能有问题，导致和角谱的中心不一致。
    %     output_phase = output_phase - min(min(output_phase));
    output_phase = output_phase.*cyl(x,y,dim/2);
%     error_OSS(t) = sqrt(sum(sum((output_phase-phase_T).^2)))./(EM.*EM);
    error_OSS(t) = sqrt(sum(sum((output_phase-phase_T).^2))/(EM^2))/(2*pi);
%     e = sqrt(sum(sum((abs(output_phase-phase_T).*cyl(x,y,dim/2)).^2))/(EM^2))
%     output_phase(253:259,253:259)
%     phase_T(253:259,253:259)
end
E_plane_output = two_step_prop_ASM(E_plane_2,lambda,delta2,delta1,-z);
% T_phase = angle(E_plane_output);
% E_plane_output = abs(E_plane_output).*exp(1j.*T_phase);
E_plane_output(cond1)=0;
error_OSS = gather(error_OSS);
