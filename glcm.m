%Программа вычисления 24-х статистических ключевых признаков
%цветных текстурных изображений (без фильтрации R, G, B, RG,RB, GB компонентов 
%Вычисление glcm и статистики для цветовых составляющих
%R,G,B (inter_channel)и для разностных составляющих RG, RB, GB, (intra_channel)
%полученных путем вычитания матриц R, B, G

clc;
clear;
%Обрабатываться будут нормализованные изображения размерностью 
%100х300 пикселей 
%1.jpg-15.jpg 
path = 'C:\Users\luppe\OneDrive\Документы\MATLAB\rice_leaf_diseases\Leaf smut';

fil=fullfile(path,'*.jpg');
d=dir(fil);
Rcon = zeros(numel(d),1);
Rcor  = zeros(numel(d),1);
Ren = zeros(numel(d),1);
Rhom = zeros(numel(d),1);
Gcon = zeros(numel(d),1);
Gcor = zeros(numel(d),1);
Gen = zeros(numel(d),1);
Ghom = zeros(numel(d),1);
Bcon = zeros(numel(d),1);
Bcor = zeros(numel(d),1);
Ben = zeros(numel(d),1);
Bhom = zeros(numel(d),1);
RGcon = zeros(numel(d),1);
RGcor = zeros(numel(d),1);
RGen = zeros(numel(d),1);
RGhom = zeros(numel(d),1);
RBcon = zeros(numel(d),1);
RBcor = zeros(numel(d),1);
RBen = zeros(numel(d),1);
RBhom = zeros(numel(d),1);
GBcon = zeros(numel(d),1);
GBcor = zeros(numel(d),1);
GBen = zeros(numel(d),1);
GBhom = zeros(numel(d),1);
for k=1:numel(d)
  filename=fullfile(path,d(k).name);
   he = imread(filename);
   figure;
imshow(he);

%выделение r,g,b компонент
r=he(:,:,1);
g=he(:,:,2);
b=he(:,:,3);

%вычисление inter_channel_matrix
rgb_image=im2double(he);%преобразование элементов изображения в формат double
%figure, imshow(he);
fR = rgb_image (:, :, 1);
fG = rgb_image (:, :, 2);
fB = rgb_image (:, :, 3);

%вычисление intra_channel_matrix fRG, fRB, fGB путем вычитания матриц
fRG=fR-fG;
fRB=fR-fB;
fGB=fG-fB;
%или по формулам
%     
%Вычисление glcm и статистики для цветовых составляющих

%R,G,B (inter_channel)
glcm = graycomatrix(fR, 'Offset',[2 0]);%вычисление матрицы glcm
stats_R = graycoprops(glcm);%вычисление статистических характеристик glcm
glcm = graycomatrix(fG, 'Offset',[2 0]);%вычисление матрицы glcm
stats_G = graycoprops(glcm);%вычисление статистических характеристик glcm
glcm = graycomatrix(fB, 'Offset',[2 0]);%вычисление матрицы glcm
stats_B = graycoprops(glcm);%вычисление статистических характеристик glcm

%Вычисление glcm и статистики для разностных составляющих RG, RB, GB, (intra_channel)
%полученных путем вычитания матриц fR, fB, fG
glcm = graycomatrix(fRG, 'Offset',[2 0]);%вычисление матрицы glcm
stats_RG = graycoprops(glcm);%вычисление статистических характеристик glcm
glcm = graycomatrix(fRB, 'Offset',[2 0]);%вычисление матрицы glcm
stats_RB = graycoprops(glcm);%вычисление статистических характеристик glcm
glcm = graycomatrix(fGB, 'Offset',[2 0]);%вычисление матрицы glcm
stats_GB = graycoprops(glcm);%вычисление статистических характеристик glcm
Rcon(k)=stats_R.Contrast;
Rcor(k)=stats_R.Correlation;
Ren(k)=stats_R.Energy;
Rhom(k)=stats_R.Homogeneity;
Gcon(k)=stats_G.Contrast;
Gcor(k)=stats_G.Correlation;
Gen(k)=stats_G.Energy;
Ghom(k)=stats_G.Homogeneity;
Bcon(k)=stats_B.Contrast;
Bcor(k)=stats_B.Correlation;
Ben(k)=stats_B.Energy;
Bhom(k)=stats_B.Homogeneity;
RGcon(k)=stats_RG.Contrast;
RGcor(k)=stats_RG.Correlation;
RGen(k)=stats_RG.Energy;
RGhom(k)=stats_RG.Homogeneity;
RBcon(k)=stats_RB.Contrast;
RBcor(k)=stats_RB.Correlation;
RBen(k)=stats_RB.Energy;
RBhom(k)=stats_RB.Homogeneity;
GBcon(k)=stats_GB.Contrast;
GBcor(k)=stats_GB.Correlation;
GBen(k)=stats_GB.Energy;
GBhom(k)=stats_GB.Homogeneity;
%A=magic(5)%для тестирования
end

t = ["Rcon", "Rcor", "Ren", "Rhom", "Gcon", "Gcor", "Gen", "Ghom", "Bcon", "Bcor", "Ben", "Bhom", "RGcon", "RGcor", "RGen", "RGhom", "RBcon", "RBcor", "RBen", "RBhom", "GBcon", "GBcor", "GBen", "GBhom"]
A=[Rcon Rcor Ren Rhom Gcon Gcor Gen Ghom Bcon Bcor Ben Bhom RGcon RGcor RGen RGhom RBcon RBcor RBen RBhom GBcon GBcor GBen GBhom]
%fid = fopen('res1D.xls', 'wb');% передача вектор-строки параметров Харалика в Excel
A_t = transpose(A)
T = table(transpose(t), A_t)
writetable(T,'Leaf_smut.xls');
%xlswrite('res1D.xlsx', {transpose(t);A});
%pause;
close all;
clear;
