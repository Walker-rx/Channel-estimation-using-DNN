function [ dlnet, velocity, losss, learnRate_save ] = dnn_train_custom(maxEpochs, numOber, xTrain, yTrain, xValidation, yValidation, ...
                                                            numIterPerEpoch, miniBatchSize, dlnet, velocity, inilearningRate, ...
                                                            momentum, train_loop_time, train_time, LearnRateDropPeriod, ...
                                                            LearnRateDropFactor, validationFrequency)
    ite = 0;
    learnRate = inilearningRate;
    losss = [];
    learnRate_save = [];
    averageGrad = [];
    averageSqGrad = [];
    gradDecay = 0.9;
    sqGradDecay = 0.999;

    valid_loss = 50000;
    valid_num = 0;
    XValidation(:,:,1) = xValidation;
    YValidation(:,:,1) = yValidation;
    dlXValidation = dlarray(single(XValidation),'CBT');
    dlYValidation = dlarray(single(YValidation),'CBT');
    dlXValidation = gpuArray(dlXValidation);
    dlYValidation = gpuArray(dlYValidation);
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
    
%             [dlnet, velocity] = sgdmupdate(dlnet,gradients,velocity,learnRate,momentum);
            [dlnet, averageGrad,averageSqGrad] = adamupdate(dlnet,gradients,averageGrad,averageSqGrad,ite,learnRate,gradDecay,sqGradDecay);
    
            losss(ite) = extractdata(loss);
            learnRate_save(ite) = learnRate;
            if mod(ite,floor(numIterPerEpoch/5)) == 0
                fprintf(" looptime = %d , training times = %d , epoches = %d , iteration = %d , loss = %e , learnRate = %e \n",...
                    train_loop_time,train_time,epoch,ite,losss(ite),learnRate);
                pause(0.5)
            end
            clear X Y dlX dlY
            
            if mod(i,validationFrequency) == 0
                [~,~,loss_validation] = dlfeval(@modelGradientss,dlnet,dlXValidation,dlYValidation);
                loss_validation = gather(extractdata(loss_validation));
                if valid_loss <= loss_validation
                    valid_num = valid_num+1;
                    fprintf(" valid num = %d , minimum loss = %e , valid num changed \n",valid_num,valid_loss);
                else
                    valid_loss = loss_validation;
                    fprintf(" valid num = %d , minimum loss = %e , valid num not changed \n",valid_num,valid_loss);
                end
            end
            if valid_num == 100
                return
            end
        end
    
        if mod(epoch,LearnRateDropPeriod) == 0
            learnRate = learnRate*LearnRateDropFactor;
        end
    end

end