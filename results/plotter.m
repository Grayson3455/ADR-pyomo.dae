clc
clear 
close all

A =  figure('Renderer', 'painters', 'Position', [200 100 800 800]);

% read the data
x = csvread('x.csv',0,0);
y = csvread('y.csv',0,0);
phi = csvread('phi.csv',0,0);

% the real solution
phi_real = @(x,y) 1/2*(1 - tanh( (x-0.5)/0.05));

%create mesh grid
[X,Y] = meshgrid(x,y);
surf(X,Y,phi)

% plotting parameters
xlabel('x','interpreter','latex')
ylabel('y','interpreter','latex')
zlabel('$\phi_h$','interpreter','latex')
set(gca,'FontSize',30)
a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times')

view(3)
colorbar()
savename = ['/Users/gxt/Desktop/My_Computer/Courses/CBE-NSO/ADR-pyomo.dae/report/fig/steady-adr-smooth.png'];
saveas(A,savename)

% calculate nodal error matrix
for i = 1: length(x)
    for j = 1: length(y)
        error(i,j) = abs(phi(i,j) - phi_real(x(i),y(j)));
    end
end

B =  figure('Renderer', 'painters', 'Position', [200 100 800 800]);
surf(X,Y,error)
colorbar()
xlabel('x','interpreter','latex')
ylabel('y','interpreter','latex')
zlabel('Error','FontName','Times')
set(gca,'FontSize',30)
a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'FontName','Times')

view(3)
colorbar()
savename = ['/Users/gxt/Desktop/My_Computer/Courses/CBE-NSO/ADR-pyomo.dae/report/fig/steady-adr-smooth-error.png'];
saveas(B,savename)
