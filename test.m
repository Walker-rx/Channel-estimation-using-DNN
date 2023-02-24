clear
close all

h_order = 6;
xTest = cell(1,10);
yTest = cell(1,10);
xTest2 = cell(1,10);
yTest2 = cell(1,10);

for i = 1:10
    xTest{i} = 1:1200;
    yTest{i} = 1:1200;
end

for i = 1:numel(xTest)
    xTest_p1 = toeplitz(xTest{i}(h_order:-1:1),xTest{i}(h_order:end));
    xTest_p2 = toeplitz([0,xTest{i}(length(xTest{i})...
        :-1:...
        length(xTest{i})-(h_order-2)...
        )...
        ],...
        zeros(1,h_order-1));
    xTest2{i} = [xTest_p1,xTest_p2];
    yTest2{i} = reshape(yTest{i},6,[]);
end

