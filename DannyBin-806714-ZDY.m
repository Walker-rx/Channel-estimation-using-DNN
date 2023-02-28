%（一）自定义神经网络
net=network;
%设置网络对象结构
net.numInputs=2;
net.numLayers=3;
net.biasConnect=[1 0 1]';
net.inputConnect=[1 0;1 1;0 0];
net.layerConnect=[0 0 0;0 0 0;1 1 1];
net.outputConnect=[0 1 1];
net.targetConnect=[0 0 1];
%设置子对象结构属性
net.inputs{1}.range=[0 2;0 2]
net.inputs{2}.range=[-2 2;-2 2;-2 2;-2 2;-2 2]
net.inputs{1}
net.inputs{2}
%网络层设置
net.layers{1}.size=4;
net.layers{1}.initFcn='initnw';
net.layers{1}.transferFcn='tansig';
net.layers{2}.size=3;
net.layers{2}.initFcn='initnw';
net.layers{2}.transferFcn='logsig';
net.layers{3}.initFcn='initnw';
%网络层
net.layers{1}
net.layers{2}
net.layers{3}
%输出向量
net.outputs{1}
net.outputs{2}
net.outputs{3}
%目标向量
net.targets{1}
net.targets{2}
net.targets{3}
%阀值向量
net.biases{1}
net.biases{2}
net.biases{3}
%输入权值向量
net.inputWeights{1,1}
net.inputWeights{1,2}
net.inputWeights{2,1}
net.inputWeights{2,2}
net.inputWeights{3,1}
net.inputWeights{3,2}
%连接权有延迟
net.inputWeights{2,1}.delays=[0 1];
net.inputWeights{2,2}.delays=1;
net.layerWeights{3,3}.delays=1;
%网络权值向量
net.layerWeights{1,1}
net.layerWeights{1,2}
net.layerWeights{1,3}
net.layerWeights{2,1}
net.layerWeights{2,2}
net.layerWeights{2,3}
net.layerWeights{3,1}
net.layerWeights{3,2}
net.layerWeights{3,3}
%设置函数属性
net.initFcn='initlay';
net.performFcn='mse';
net.trainFcn='trainlm';
%参数属性
net.IW
net.LW
net.b
%（二）神经网络初始化
net=init(net)
net.IW
net.IW{1,1}
net.IW{2,1}
net.IW{2,2}
net.LW
net.LW{3,1}
net.LW{3,2}
net.LW{3,3}
net.b
net.b{1}
net.b{3}
%(三)网络训练
p={[0;0] [2;0.5];[2;-2;1;0;1] [-1;-1;1;0;1]};
t={1 -1};
net=train(net,p,t);
%（四）仿真
p={[0;0] [2;0.5];[2;-2;1;0;1] [-1;-1;1;0;1]};
y=sim(net,p)
%仿真结果
y{1,1}
y{1,2}%第一个网络输出
y2={ y{2,1},y{2,2}}%第二个网络输出（第3个网络层的输出）