function [ dlnet, velocity, losss, learnRate_save ] = dnn_train_custom(maxEpochs, numOber, xTrain, yTrain, numIterPerEpoch, miniBatchSize, dlnet, ...
        velocity, inilearningRate, momentum,train_loop_time, train_time, LearnRateDropPeriod, LearnRateDropFactor)
    ite = 0;
    learnRate = inilearningRate;
    losss = [];
    learnRate_save = [];
    for epoch = 1:maxEpochs
        idx = randperm(numOber);
        xTrain = xTrain(:,idx);
        yTrain = yTrain(:,idx);
    
        for i = 1:numIterPerEpoch
            ite = ite + 1;
            idx = (i-1)*miniBatchSize+1 : miniBatchSize*i;
            X(:,:,1) = xTrain(:,idx);
            Y(:,:,1) = yTrain(:,idx);
            dlX = dlarray(single(X),'CBT');
            dlY = dlarray(single(Y),'CBT');
            dlX = gpuArray(dlX);
            dlY = gpuArray(dlY);
    
            [gradients,state,loss] = dlfeval(@modelGradientss,dlnet,dlX,dlY);
            dlnet.State = state;
    
            [dlnet, velocity] = sgdmupdate(dlnet,gradients,velocity,learnRate,momentum);
    
            losss(ite) = extractdata(loss);
            learnRate_save(ite) = learnRate;
            if mod(ite,floor(numIterPerEpoch/2)) == 0
                fprintf(" looptime = %d , training times = %d , epoches = %d , iteration = %d , loss = %e , learnRate = %e \n",...
                    train_loop_time,train_time,epoch,ite,losss(ite),learnRate);
            end
            clear X Y dlX dlY
        end
    
        if mod(epoch,LearnRateDropPeriod) == 0
            learnRate = learnRate*LearnRateDropFactor;
        end
    end

end