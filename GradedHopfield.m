clear

% open figure
figure(1)
set(gcf, 'position', [50 450 2100 600])

% input current vs. activity
c = 10;
g = @(x) 1 ./ (exp(-c*(x - 1/2)) + 1);
inv_g = @(x) -1/c*log(1./x - 1) + 1/2;

x = [0:0.01:1];
y = g(x);

subplot(1,3,1)
h = plot(x, y);
set(h, 'linewidth', 5)
xlabel('input current')
ylabel('activity')
set(gca, 'linewidth', 3, 'fontsize', 25)
axis([0 1 0 1])

% two neuron simulation
subplot(1,3,2)

a = -0.8;
b = 0.75;
tau = 1;
T = [0 a; a 0];
I = [b b]';
f = @(t, x) -x/tau + T*g(x) + I;

isFirst = true;
for v1 = [0.1:0.1:0.9]
    for v2 = [0.09:0.1:0.9]
        x0 = inv_g([v1, v2]');
        [t, x] = ode23(f, [0, 10], x0);
        V = g(x);
        V1 = V(:,1);
        V2 = V(:,2);

        if isFirst, hold off, isFirst = false;
        else hold on
        end
        h = plot(V1, V2, 'k');
        set(h, 'linewidth', 1)
        plot(V1(end), V2(end), 'ro');
    end
end

xlabel('neuron 1 activity')
ylabel('neuron 2 activity')
set(gca, 'linewidth', 3, 'fontsize', 25)
axis([0 1 0 1])

% energy
subplot(1,3,3)
int_inv_g = @(x) -1/10*(x.*log(1./x - 1) - log(1 - x)) + 1/2*x;
m = [0.01:0.02:1-0.01];
[V1, V2] = meshgrid(m, m);
E = -1/2*(2*T(1,2)*V1.*V2) - (I(1)*V1 + I(2)*V2) + 1/tau*(int_inv_g(V1) + int_inv_g(V2));
contour(V1, V2, E, -1:0.001:1)
colormap('rainbow')
xlabel('neuron 1 activity')
ylabel('neuron 2 activity')
set(gca, 'linewidth', 3, 'fontsize', 25)
axis([0 1 0 1])
