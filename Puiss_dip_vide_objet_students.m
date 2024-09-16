%calcul puiss dipole vide 
%09/09/2024, nouveau TP, 3D
% ATTENTION AUX UNITES :


clear all
close all

racine='';
dl=num2str(20);%taille cellules nm
dS=(str2num(dl)*1e-9)^2; %en m^2
n=1;
lambda=0.8e-6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=2.99792458e8;
mu0=4*pi*1e-7;
eps0=1/(c^2*mu0);

eps=n^2;
Z0=sqrt(mu0/eps0);
i=sqrt(-1);

%--------------------
w=2*pi*c/lambda;
k=2*pi*n/lambda; %k=k0*n

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Xplus

fid=['dip_vide_',dl,'nm',racine,'_Xplus_f1_ex.dat'];
EEx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xplus_f1_ey.dat'];
EEy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xplus_f1_ez.dat'];
EEz=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xplus_f1_hx.dat'];
HHx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xplus_f1_hy.dat'];
HHy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xplus_f1_hz.dat'];
HHz=leerdatos(fid,3);

% radio_PML_touch
cc=0;
for jj=1:2:size(EEy,2)
    cc=cc+1;
    abs_Ex(:,cc)=EEx(:,jj);
    angle_Ex(:,cc)=EEx(:,jj+1);
    
    abs_Ey(:,cc)=EEy(:,jj);
    angle_Ey(:,cc)=EEy(:,jj+1);
    
    abs_Ez(:,cc)=EEz(:,jj);
    angle_Ez(:,cc)=EEz(:,jj+1);
    
    abs_Hx(:,cc)=HHx(:,jj);
    angle_Hx(:,cc)=HHx(:,jj+1);
    
    abs_Hy(:,cc)=HHy(:,jj);
    angle_Hy(:,cc)=HHy(:,jj+1);
    
    abs_Hz(:,cc)=HHz(:,jj);
    angle_Hz(:,cc)=HHz(:,jj+1);
end

Ex=abs_Ex.*exp(i*angle_Ex/180*pi);%figure;imagesc(abs(Ex));colorbar;axis square
Ey=abs_Ey.*exp(i*angle_Ey/180*pi);
Ez=abs_Ez.*exp(i*angle_Ez/180*pi);

Hx=abs_Hx.*exp(i*angle_Hx/180*pi);
Hy=abs_Hy.*exp(i*angle_Hy/180*pi);
Hz=abs_Hz.*exp(i*angle_Hz/180*pi);%figure;imagesc(abs(Hz));colorbar;axis square


Puiss_Xplus=0.5*dS/Z0*1e12*sum(sum(real(Ey.*conj(Hz)-Ez.*conj(Hy))))

clear Ex Ey Ez Hx Hy Hz 
%---------------------------------------------
%Xmoins

fid=['dip_vide_',dl,'nm',racine,'_Xmoins_f1_ex.dat'];
EEx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xmoins_f1_ey.dat'];
EEy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xmoins_f1_ez.dat'];
EEz=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xmoins_f1_hx.dat'];
HHx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xmoins_f1_hy.dat'];
HHy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Xmoins_f1_hz.dat'];
HHz=leerdatos(fid,3);

% radio_PML_touch
cc=0;
for jj=1:2:size(EEy,2)
    cc=cc+1;
    abs_Ex(:,cc)=EEx(:,jj);
    angle_Ex(:,cc)=EEx(:,jj+1);
    
    abs_Ey(:,cc)=EEy(:,jj);
    angle_Ey(:,cc)=EEy(:,jj+1);
    
    abs_Ez(:,cc)=EEz(:,jj);
    angle_Ez(:,cc)=EEz(:,jj+1);
    
    abs_Hx(:,cc)=HHx(:,jj);
    angle_Hx(:,cc)=HHx(:,jj+1);
    
    abs_Hy(:,cc)=HHy(:,jj);
    angle_Hy(:,cc)=HHy(:,jj+1);
    
    abs_Hz(:,cc)=HHz(:,jj);
    angle_Hz(:,cc)=HHz(:,jj+1);
end

Ex=abs_Ex.*exp(i*angle_Ex/180*pi);
Ey=abs_Ey.*exp(i*angle_Ey/180*pi);
Ez=abs_Ez.*exp(i*angle_Ez/180*pi);

Hx=abs_Hx.*exp(i*angle_Hx/180*pi);
Hy=abs_Hy.*exp(i*angle_Hy/180*pi);
Hz=abs_Hz.*exp(i*angle_Hz/180*pi);


Puiss_Xmoins=-0.5*dS/Z0*1e12*sum(sum(real(Ey.*conj(Hz)-Ez.*conj(Hy))))


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Yplus

fid=['dip_vide_',dl,'nm',racine,'_Yplus_f1_ex.dat'];
EEx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Yplus_f1_ey.dat'];
EEy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Yplus_f1_ez.dat'];
EEz=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Yplus_f1_hx.dat'];
HHx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Yplus_f1_hy.dat'];
HHy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Yplus_f1_hz.dat'];
HHz=leerdatos(fid,3);

% radio_PML_touch
cc=0;
for jj=1:2:size(EEy,2)
    cc=cc+1;
    abs_Ex(:,cc)=EEx(:,jj);
    angle_Ex(:,cc)=EEx(:,jj+1);
    
    abs_Ey(:,cc)=EEy(:,jj);
    angle_Ey(:,cc)=EEy(:,jj+1);
    
    abs_Ez(:,cc)=EEz(:,jj);
    angle_Ez(:,cc)=EEz(:,jj+1);
    
    abs_Hx(:,cc)=HHx(:,jj);
    angle_Hx(:,cc)=HHx(:,jj+1);
    
    abs_Hy(:,cc)=HHy(:,jj);
    angle_Hy(:,cc)=HHy(:,jj+1);
    
    abs_Hz(:,cc)=HHz(:,jj);
    angle_Hz(:,cc)=HHz(:,jj+1);
end

Ex=abs_Ex.*exp(i*angle_Ex/180*pi);
Ey=abs_Ey.*exp(i*angle_Ey/180*pi);
Ez=abs_Ez.*exp(i*angle_Ez/180*pi);

Hx=abs_Hx.*exp(i*angle_Hx/180*pi);
Hy=abs_Hy.*exp(i*angle_Hy/180*pi);
Hz=abs_Hz.*exp(i*angle_Hz/180*pi);


Puiss_Yplus=0.5*dS/Z0*1e12*sum(sum(real(Ez.*conj(Hx)-Ex.*conj(Hz))))


%---------------------------------------------
%Ymoins

fid=['dip_vide_',dl,'nm',racine,'_Ymoins_f1_ex.dat'];
EEx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Ymoins_f1_ey.dat'];
EEy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Ymoins_f1_ez.dat'];
EEz=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Ymoins_f1_hx.dat'];
HHx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Ymoins_f1_hy.dat'];
HHy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Ymoins_f1_hz.dat'];
HHz=leerdatos(fid,3);

% radio_PML_touch
cc=0;
for jj=1:2:size(EEy,2)
    cc=cc+1;
    abs_Ex(:,cc)=EEx(:,jj);
    angle_Ex(:,cc)=EEx(:,jj+1);
    
    abs_Ey(:,cc)=EEy(:,jj);
    angle_Ey(:,cc)=EEy(:,jj+1);
    
    abs_Ez(:,cc)=EEz(:,jj);
    angle_Ez(:,cc)=EEz(:,jj+1);
    
    abs_Hx(:,cc)=HHx(:,jj);
    angle_Hx(:,cc)=HHx(:,jj+1);
    
    abs_Hy(:,cc)=HHy(:,jj);
    angle_Hy(:,cc)=HHy(:,jj+1);
    
    abs_Hz(:,cc)=HHz(:,jj);
    angle_Hz(:,cc)=HHz(:,jj+1);
end

Ex=abs_Ex.*exp(i*angle_Ex/180*pi);
Ey=abs_Ey.*exp(i*angle_Ey/180*pi);
Ez=abs_Ez.*exp(i*angle_Ez/180*pi);

Hx=abs_Hx.*exp(i*angle_Hx/180*pi);
Hy=abs_Hy.*exp(i*angle_Hy/180*pi);
Hz=abs_Hz.*exp(i*angle_Hz/180*pi);


Puiss_Ymoins=-0.5*dS/Z0*1e12*sum(sum(real(Ez.*conj(Hx)-Ex.*conj(Hz))))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Zplus

fid=['dip_vide_',dl,'nm',racine,'_Zplus_f1_ex.dat'];
EEx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zplus_f1_ey.dat'];
EEy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zplus_f1_ez.dat'];
EEz=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zplus_f1_hx.dat'];
HHx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zplus_f1_hy.dat'];
HHy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zplus_f1_hz.dat'];
HHz=leerdatos(fid,3);

% radio_PML_touch
cc=0;
for jj=1:2:size(EEy,2)
    cc=cc+1;
    abs_Ex(:,cc)=EEx(:,jj);
    angle_Ex(:,cc)=EEx(:,jj+1);
    
    abs_Ey(:,cc)=EEy(:,jj);
    angle_Ey(:,cc)=EEy(:,jj+1);
    
    abs_Ez(:,cc)=EEz(:,jj);
    angle_Ez(:,cc)=EEz(:,jj+1);
    
    abs_Hx(:,cc)=HHx(:,jj);
    angle_Hx(:,cc)=HHx(:,jj+1);
    
    abs_Hy(:,cc)=HHy(:,jj);
    angle_Hy(:,cc)=HHy(:,jj+1);
    
    abs_Hz(:,cc)=HHz(:,jj);
    angle_Hz(:,cc)=HHz(:,jj+1);
end

Ex=abs_Ex.*exp(i*angle_Ex/180*pi);
Ey=abs_Ey.*exp(i*angle_Ey/180*pi);
Ez=abs_Ez.*exp(i*angle_Ez/180*pi);

Hx=abs_Hx.*exp(i*angle_Hx/180*pi);
Hy=abs_Hy.*exp(i*angle_Hy/180*pi);
Hz=abs_Hz.*exp(i*angle_Hz/180*pi);


Puiss_Zplus=0.5*dS/Z0*1e12*sum(sum(real(Ex.*conj(Hy)-Ey.*conj(Hx))))


%---------------------------------------------
%Zmoins

fid=['dip_vide_',dl,'nm',racine,'_Zmoins_f1_ex.dat'];
EEx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zmoins_f1_ey.dat'];
EEy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zmoins_f1_ez.dat'];
EEz=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zmoins_f1_hx.dat'];
HHx=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zmoins_f1_hy.dat'];
HHy=leerdatos(fid,3);

fid=['dip_vide_',dl,'nm',racine,'_Zmoins_f1_hz.dat'];
HHz=leerdatos(fid,3);

% radio_PML_touch
cc=0;
for jj=1:2:size(EEy,2)
    cc=cc+1;
    abs_Ex(:,cc)=EEx(:,jj);
    angle_Ex(:,cc)=EEx(:,jj+1);
    
    abs_Ey(:,cc)=EEy(:,jj);
    angle_Ey(:,cc)=EEy(:,jj+1);
    
    abs_Ez(:,cc)=EEz(:,jj);
    angle_Ez(:,cc)=EEz(:,jj+1);
    
    abs_Hx(:,cc)=HHx(:,jj);
    angle_Hx(:,cc)=HHx(:,jj+1);
    
    abs_Hy(:,cc)=HHy(:,jj);
    angle_Hy(:,cc)=HHy(:,jj+1);
    
    abs_Hz(:,cc)=HHz(:,jj);
    angle_Hz(:,cc)=HHz(:,jj+1);
end

Ex=abs_Ex.*exp(i*angle_Ex/180*pi);
Ey=abs_Ey.*exp(i*angle_Ey/180*pi);
Ez=abs_Ez.*exp(i*angle_Ez/180*pi);

Hx=abs_Hx.*exp(i*angle_Hx/180*pi);
Hy=abs_Hy.*exp(i*angle_Hy/180*pi);
Hz=abs_Hz.*exp(i*angle_Hz/180*pi);


Puiss_Zmoins=-0.5*dS/Z0*1e12*sum(sum(real(Ex.*conj(Hy)-Ey.*conj(Hx))))


P0=Puiss_Zplus+Puiss_Yplus+Puiss_Xplus+Puiss_Zmoins+Puiss_Ymoins+Puiss_Xmoins

