function [gradients,state,loss] = modelGradients(dlnet,dlY,dlX1,dlX2)%,dlY)%dlX3,dlY)
	[dlYPred, state] = forward(dlnet,dlX1,dlX2);
	loss = mse(dlYPred,dlY);
	gradients = dlgradient(loss,dlnet.Learnables);
end

