clc,clear
filename = 'E:\00_PhaseRetrieval\PhENN\dataset\train\intensity\';
num = 100;
I = zeros(512, 512, 100);
for index = 1:num
    load([filename 'image' num2str(index,'%06d') '.mat']);
    I(:,:,index) = Iz;
end
figure(1),
histogram(Iz)