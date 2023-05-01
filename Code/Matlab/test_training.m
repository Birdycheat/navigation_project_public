%% clear all and load data

clearvars
close all
clc

Files1 = dir(fullfile('AG', '*.mat'));
Files2 = dir(fullfile('GB', '*.mat'));
Files3 = dir(fullfile('MD', '*.mat'));
Files = [Files1; Files2; Files3];

XTrain = cell(0);
YTrain = cell(0);

Files_test = dir(fullfile("data_test", "*.mat"));
XTest = cell(0);
YTest = cell(0);

iteration = 0;
index_train = 0;
index_test = 0;

Nwindows = 10;
windows = 80 + randi(220, 1, Nwindows);

numFeatures = 9;
% Train data
for i=1:length(Files)
    iteration = iteration + 1;
    filename = Files(i).name;
    inter = split(filename,'_');
    cat = inter{1};
    load(strcat(Files(i).folder, '\', Files(i).name))
    
    for window_index=1:length(windows)
        numel_indexes = floor(size(x, 2)/windows(window_index));
        reste = size(x, 2) - numel_indexes*windows(window_index);
        mat = mat2cell(x,numFeatures,[windows(window_index)*ones(1,numel_indexes) reste]);
        
        
        for k = 1:length(mat)
            if size(mat{1, k}, 2) ~= 0
                A = string(zeros(1,size(mat{1, k}, 2)));
                A(:) = cat;
                XTrain{end+1,1} = mat{1, k};
                YTrain{end+1,1} = categorical(A);
            end
        end
    end
end

% Test data
for i=1:length(Files_test)
    iteration = iteration + 1;
    filename = Files_test(i).name;
    inter = split(filename,'_');
    cat = inter{1};
    load(strcat(Files_test(i).folder, '\', Files_test(i).name))
        

    A = string(zeros(1,size(x, 2)));
    A(:) = cat;
    XTest{end+1,1} = x;
    YTest{end+1,1} = categorical(A);
end

%%
numHiddenUnits = 500;
numClasses = 3;

layers = [ ...
    sequenceInputLayer(numFeatures)
    batchNormalizationLayer
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);


%% Test classification
total = 0;
correct = 0;
display = true;

for i=1:length(YTest)
    YPred = classify(net,XTest{i});
    correct = correct + sum(YPred == YTest{i});
    total = total + length(YTest{i});
    if display == true
        figure
        plot(YPred,'.-')
        hold on
        plot(YTest{i})
        hold off
        
        xlabel("Time Step")
        ylabel("Activity")
        title("Predicted Activities")
        legend(["Predicted" "Test Data"])
        pause(1)
    end
end
fprintf("Test Accuracy = %f pourcents\n", correct/total*100)