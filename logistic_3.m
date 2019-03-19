% 使用自己写的logistic回归的损失函数f_cost，用fminunc求最小值
% 数据是自己编的
%% 导入数据并全局化
X0 = xlsread('T:\matlab代码\data_gra.xlsx', 'B2:D301');
Y = xlsread('T:\matlab代码\data_gra.xlsx', 'E2:E301');
X = [ones(size(X0, 1), 1) X0]; %调整后X
global X; 
global Y;

%% 求局部最优解并得到预测值
[w minv] = fminunc(@f_cost, [0 0 0 0]');

%获得参数之后构建概率函数
f_pi = @(x) exp(x' * w) ./ (1 + exp(x' * w));  
Y_prd = zeros(size(X, 1), 1);
Y_prd(f_pi(X') >= 0.5) = 1;
%注意，f_pi使用./后拓展性增强了，对于一般的列向量输入，结果和以前一样，
%改变之后可以适应输入为矩阵的情况，保证输入矩阵的每一列为一个输入，故X转置
%原来的：
%f_pi = @(x) exp(x' * w) / (1 + exp(x' * w)); 
% Y_prd = zeros(size(X, 1), 1);
% for i = 1 : size(X, 1)
%     if f_pi(X(i, :)') >= 0.5
%         Y_prd(i) = 1;
%     end
% end

Y_cmp = [Y Y_prd]; %原分类与预测值合并比较

%% 定义logistic损失函数
function value = f_cost(w)

global X;
global Y;
m = size(X, 1);
value = -1 * ones(1, m) * (Y .* (X * w) - log(1 + exp(X * w)));
% 使用Y ./ (...)利用点乘可以使代码简洁，替代以下5行
%并且貌似效率更高，因为得到的最小值更小，然而fminunc说没有满足最小限度
% value = 0;
% for i = 1 : m
%     value = value + Y(i) * (X(i, :) * w) - log(1 + exp(X(i,:) * w));
% end
% value = -value;

end
