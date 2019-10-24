function Z = zernike (i, r, theta)  
% i表示泽尼克多项式的第i项，r,theta分别表示极坐标的两个坐标轴（半径坐标和角坐标）
% 'zernike index'指的是泽尼克多项式每一项指数m,n的列表

load ('zernike_index'); 
n = zer_index(i,1);
m = zer_index(i,2);
if m == 0
    Z = sqrt(n+1)*zrf(n,0,r);
else 
    if mod(i,2) == 0 
        Z = sqrt(2*(n+1))*zrf(n,m,r).*cos(m*theta);
    else             
        Z = sqrt(2*(n+1))*zrf(n,m,r).*sin(m*theta);
    end 
end
return

% Zernike radial function
function R = zrf(n, m, r)
R = 0;
for s = 0:((n-m)/2)
    num = (-1)^s*gamma(n-s+1);
    denom = gamma(s+1)*gamma((n+m)/2-s+1)*gamma((n-m)/2-s+1);
    R = R+num/denom*r.^(n-2*s);
end



