function z=cyl(x,y,r0)
error(nargchk(2,3,nargin));
if nargin<3,r0=1; end
r=sqrt(x.*x+y.*y);
z=(r<r0)*1.0;
z(find(r==r0))=0.5;