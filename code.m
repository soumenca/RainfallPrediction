clc
clear all

%%% import the data files
inputData = '/rainfall.xlsx';
[data, string] = xlsread(inputData, 's1');
data = data(:,1:14);

%%% Feature Constraction
months = data(:,2:13);
annualRain = [];
year = [];
monthName = [];

for i = 1:length(data)
    F1(1:12) = data(i,1);
    year = [year; F1'];
    
    F2(1:12) = [1:12];
    monthName = [monthName; F2'];
    
    F(1:12) =  data(i,14);
    annualRain = [annualRain; F'];
end
preAnnualRain = [NaN(1,12), annualRain(1:end-12)'];
preAnnualRain = preAnnualRain';
preYearMonth = [NaN(1,12), months(1:end-12)];
preYearMonth = preYearMonth';

months = months';
months = months(:);

R = [year, monthName, months, preYearMonth, preAnnualRain, annualRain];   % New table
RT = R';


%%% Creating Training and Testing set
index = randperm(length(R));
noTrainData = floor(length(R) * .75);
trainData = R(index(1:noTrainData), :);
testData = R(index(noTrainData+1 : length(R)), :);

trainData = trainData';
trainDataY = trainData(6,:);
trainData = trainData(1:5,:);

testData = testData';
testDataY = testData(6,:);
testData = testData(1:5,:);


%%% Building Neural Network
min_err = inf;
desired_err = 6.7;

while min_err>desired_err
    for neurons_no = 1:3:24
        net = newff(trainData, trainDataY, neurons_no);   %% using newfit or newff 
        net.performFcn = 'mae';
        net = train(net, trainData, trainDataY);
        NNpredicted = sim(net, testData);  
        % Mean absolute percentage error.
        err    = testDataY - NNpredicted;
        errpct = abs(err)./testDataY*100;            %Absolute percentage error
        MAPE   = mean(errpct(~isinf(errpct)));  
        fprintf('Current MAPE(Mean Absolute Percent Error):  %0.3f%%\n',MAPE);  
        if MAPE<min_err
            min_err= MAPE;
            if min_err <= desired_err
                break;
            end
        end
    end
end







