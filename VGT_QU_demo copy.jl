using PyCall,PyPlot,LsqFit,Images,LinearAlgebra,StatsBase,FITSIO,HDF5,Statistics
using LazCore,LazCyvecd,LazInstaller,LazIO,LazType,LazVGT
@pyimport numpy as np
@pyimport glob as glob

#commands for intalling package:
#using Pkg;
#Pkg.add("name of package")


#The path of external packages beginging with Laz... should be included to the LOADPATH
#############################
#open FITS data set:
f = FITS("/Users/p.akhilesh/Desktop/Laz/M33_14B-088_HI.clean.image.GBT_feathered.2.6kms.fits");
#assign the 3D PPV cube to d;
d = read(f[1]);
nx,ny,nv = size(d);

#read header
header = read_header(f[1]);
#calculate coordinate and line-of-sight velocity
RA,DEC,vz = getcoor(header,nx,ny,nv);
#velocity resolution
dv = (vz[2]-vz[1])/1000; #unit: m/s
#caclulate noise level 
noi = std(d[:,:,1][.~isnan.(d[:,:,1])])

###########################constructing Pseudo Stokes parameters
dn = 20;#sub-block size
vi = -10000; #velocity of initial channel, m/s
vf = 15000; #velocity of final channel, m/s

### You are creating channels which are basically indices starting from 1 and go on. So you would have Channel 1, Channel 2, Channel 3... In the language of computers we treat these channels like indices (maybe lists for intitutive understanding). Therefore, when the for loop of the VGT_mv_errorr runs through Ni to Nf, it treats them as indices and does the necessary calculations. 
Ni = Int(round(abs((vi-vz[1])/dv)))+1; #inital channel
Nf = Int(round(abs((vf-vz[1])/dv)))+1; #end channel
#calculating Pseudo-Stokes parameters

Ni = 1
Nf = 50

Qi,Ui, Qie, Uie=VGT_QU_mv_error(d,dn,Ni,Nf,noi);
#Qie and Uie are the uncertainties.
#this function outputs of Q, U cubes, instead of maps.
#pixels with noise larger than three sigma levels are blanked out

Qa=sum(Qi;dims=3)[:,:,1];#project the Q cube into Q map 
Ua=sum(Ui;dims=3)[:,:,1];#project the Q cube into Q map 
Qa[Qa.==0].=NaN;Ua[Ua.==0].=NaN; #Qa=0, Ua=0 means intensity is less then 3 times signal to noise ratio.
ker = 4;
# the smoothing width is FWHM = ker*2.355
Qb=imfilter_gaussian(Qa,ker); #Gaussian smoothing the map: FWHM=ker*2.355 pixels
Ub=imfilter_gaussian(Ua,ker);
psi=0.5.*atan.(Ub,Qb);
# rotate the polarization vectors by 90 degree to indicate the magnetic field direction;
psi.+=pi/2;

# decreasing the resolution of psi for visulization purpose
dnn=8;
phi=avB2dx(cos.(psi),sin.(psi),dnn);

#calculate the spectrum and find its peak
p=getpeak(d);
peak=findmax(p)[2];
#calculate the intensity maps I, velocity centroid map C, and velocity channel map Ch
d[d.<3*noi].=0;

Ni = 1
Nf = 2
Ii,C,Ch=getmap(d[:,:,Ni:Nf],vz[Ni:Nf],peak);

nx,ny=size(Ii);
Xd,Yd=np.meshgrid(div(dnn,2)+1:dnn:div(dnn,2)+ny,div(dnn,2)+1:dnn:div(dnn,2)+nx);

# supuerimpose the gradient and magnetic field vectors on the same intensity map;
# note Julia uses the Cartesian conventation for definiing angle, i.e., the angle is from right to top anticlockwisely 
figure(tight_layout="true")
imshow(Ii'.*abs(dv),origin="lower", cmap="Greys")
cb=colorbar(pad=0)
cb.ax.tick_params(labelsize=20,direction="in")
clim(0,75)
cb.set_label(label="Intensity [km/s]",size=10);
xlabel("R.A.(J2000) [degree]",size=10);
ylabel("Dec.(J2000) [degree]",size=10);
tick_params(direction="in",labelsize=10);
xticks([0,nx/4,nx*2/4,nx*3/4,nx-1],[round(RA[1]),round(RA[div(nx,4)]),round(RA[div(nx,4)*2]),round(RA[div(nx,4)*3]),round(RA[end])],size=10);
yticks([0,ny/4,ny*2/4,ny*3/4,ny-1],[round(DEC[1]),round(DEC[div(nx,4)]),round(DEC[div(nx,4)*2]),round(DEC[div(nx,4)*3]),round(DEC[end])],size=10);
quiver(Yd[1:end-1,1:end-1],Xd[1:end-1,1:end-1],cos.(phi),sin.(phi),headwidth=0,scale=40,color="r")
quiver(Yd[1:end-1,1:end-1],Xd[1:end-1,1:end-1],-cos.(phi),-sin.(phi),headwidth=0,scale=40,color="r")
title("L1551 12CO VGT")

#calculating uncertainties psie
Qe[isnan.(Qe)].=0;Ue[isnan.(Ue)].=0;
Qex=sqrt.(Proj(Qe.^2,3));
Uex=sqrt.(Proj(Ue.^2,3));

UQ=Ub./Qb;
UQe=abs.(UQ).*sqrt.((Qex./Qb).^2.0.+(Uex./Ub).^2.0);
psie=0.5.*UQe./(1.0.+UQ.^2.0);
