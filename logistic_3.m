% ʹ���Լ�д��logistic�ع����ʧ����f_cost����fminunc����Сֵ
% �������Լ����
%% �������ݲ�ȫ�ֻ�
X0 = xlsread('T:\matlab����\data_gra.xlsx', 'B2:D301');
Y = xlsread('T:\matlab����\data_gra.xlsx', 'E2:E301');
X = [ones(size(X0, 1), 1) X0]; %������X
global X; 
global Y;

%% ��ֲ����ŽⲢ�õ�Ԥ��ֵ
[w minv] = fminunc(@f_cost, [0 0 0 0]');

%��ò���֮�󹹽����ʺ���
f_pi = @(x) exp(x' * w) ./ (1 + exp(x' * w));  
Y_prd = zeros(size(X, 1), 1);
Y_prd(f_pi(X') >= 0.5) = 1;
%ע�⣬f_piʹ��./����չ����ǿ�ˣ�����һ������������룬�������ǰһ����
%�ı�֮�������Ӧ����Ϊ������������֤��������ÿһ��Ϊһ�����룬��Xת��
%ԭ���ģ�
%f_pi = @(x) exp(x' * w) / (1 + exp(x' * w)); 
% Y_prd = zeros(size(X, 1), 1);
% for i = 1 : size(X, 1)
%     if f_pi(X(i, :)') >= 0.5
%         Y_prd(i) = 1;
%     end
% end

Y_cmp = [Y Y_prd]; %ԭ������Ԥ��ֵ�ϲ��Ƚ�

%% ����logistic��ʧ����
function value = f_cost(w)

global X;
global Y;
m = size(X, 1);
value = -1 * ones(1, m) * (Y .* (X * w) - log(1 + exp(X * w)));
% ʹ��Y ./ (...)���õ�˿���ʹ�����࣬�������5��
%����ò��Ч�ʸ��ߣ���Ϊ�õ�����Сֵ��С��Ȼ��fminunc˵û��������С�޶�
% value = 0;
% for i = 1 : m
%     value = value + Y(i) * (X(i, :) * w) - log(1 + exp(X(i,:) * w));
% end
% value = -value;

end
