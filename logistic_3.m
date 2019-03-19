%% 导入数据并全局化
X = xlsread('T:\matlab代码\data_gra.xlsx', 'B2:D301');
Y = xlsread('T:\matlab代码\data_gra.xlsx', 'E2:E301');
X = [ones(size(X, 1), 1) X];
[m n] = size(X);
global X;
global Y;
global m;
global n;

%% 求局部最优解并得到预测值
w = fminunc(@f0, [0 0 0 0]');
fj = @(x) exp(x'*w) / (1 + exp(x'*w));

for i = 1 : m
if fj(X(i, :)') >= 0.5
Y1(i) = 1;
else
Y1(i) = 0;
end
end

Y2 = [Y Y1']; %原分类与预测值

%% 定义logistic损失函数
function value = f0(w)

global X;
global Y;
global m;
global n;
value = 0;
for i = 1 : size(X, 1)
    value = value + Y(i) * (X(i, :) * w) - log(1 + exp(X(i,:) * w));
end
value = -value;
end
