using PyCall,PyPlot,LsqFit,Images,LinearAlgebra,StatsBase,FFTW,FITSIO,HDF5,Statistics
using LazCore,LazCyvecd,LazInstaller,LazIO,LazType,GalToEqr
@pyimport numpy as np
@pyimport glob as glob

#read RA, DEC from GALFA_HI

#comparion with Planck polarization
f1=h5open("e:/Demo_planck2018_10arcmin.h5");
Ip=read(f1,"I");
Qp=read(f1,"Q");
Up=read(f1,"U");
#calculating the off-set angle between Galactic coordinates and equatorial coordinate 
off=convert_vec_Gal_to_Eq2000(RA[1],RA[end],DEC[1],DEC[end],nx,ny);
phi=0.5.*atan.(-Up,Qp);
phi=phi.+off.*pi./180;
#the output pna is the magnetic field angle
#we don't need to rotate the angle by pi/2 here
#the definition of planck (-U, Q) is in IAU convention
#the definition of angle in Julia is typical Cart

dnn=4;
pna=avB2dx(cos.(phi),sin.(phi),dnn);

nx,ny=size(Ip);
Xd,Yd=np.meshgrid(div(dnn,2)+1:dnn:div(dnn,2)+ny,div(dnn,2)+1:dnn:div(dnn,2)+nx);

# supuerimpose the gradient and magnetic field vectors on the same intensity map;
# note Julia uses the Cartesian conventation for definiing angle, i.e., the angle is from right to top anticlockwisely 
figure()
imshow(Ip',origin="xy", cmap="Greys")
colorbar(pad=0);
tick_params(direction="in");
xticks([0,100,200,300,400,500],[string(round(RA[1];digits=2)),string(round(RA[101];digits=2)),string(round(RA[201];digits=2)),string(round(RA[301];digits=2)),string(round(RA[401];digits=2)),string(round(RA[501];digits=2))]);
yticks([0,100,200,300,400,500],[string(round(DEC[1];digits=2)),string(round(DEC[101];digits=2)),string(round(DEC[201];digits=2)),string(round(DEC[301];digits=2)),string(round(RA[401];digits=2)),string(round(RA[501];digits=2))]);
quiver(Yd,Xd,cos.(pna),sin.(pna),headwidth=0,scale=30,color="b")
quiver(Yd,Xd,-cos.(pna),-sin.(pna),headwidth=0,scale=30,color="b")
title("Planck_demo")

