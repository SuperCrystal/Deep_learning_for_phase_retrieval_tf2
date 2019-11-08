clc,clear
filename = 'E:\00_PhaseRetrieval\PhENN\dataset\train\intensity\';
num = 2;
I = zeros(512, 512, 100);
for index = 1:num
    load([filename 'image' num2str(index,'%06d') '.mat']);
    figure,imshow(Iz)
    I(:,:,index) = Iz;
end
% figure(1),
% histogram(Iz)
% load('E:\00_PhaseRetrieval\PhENN\dataset\train\intensity\image_000001.mat');
% I1 = Iz;
% load('E:\00_PhaseRetrieval\PhENN\dataset\validate\intensity\image_000001.mat');
