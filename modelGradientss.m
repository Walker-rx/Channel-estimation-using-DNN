function [gradients,state,loss] = modelGradientss(dlnet,dlX,dlY)
	[dlYPred, state] = forward(dlnet,dlX);
	loss = mse(dlYPred,dlY);
	gradients = dlgradient(loss,dlnet.Learnables);
end

