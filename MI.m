
close all
clear
clc

file=importdata('MIdata.mat');

numCh=26;
ch_1=17;%c4 left hand
ch_2=13;%c3 right hand
ch_3=15;%cz feet
channels={'Fp1','FpZ','Fp2','Fz','FC5','FC3','FC1','FCz','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6','Pz'};
fs=256;
batchsize = 12*fs;%12s
color_1=[255,55,151]/255;
color_2=[141,221,0]/255;

x=find(isnan(file.labels));
file.labels(x)=5*ones(size(x));%change NaN to 5

count = 0;%number of all tests
startpoint1 = zeros(size(file.labels,1),2);%501760*2
startpoint2 = zeros(size(file.labels,1),2);
startpoint3 = zeros(size(file.labels,1),2);
startpoint4 = zeros(size(file.labels,1),2);
startpoint5 = zeros(size(file.labels,1),2);
for i = 2:size(file.labels,1)
    if(file.labels(i-1)==0 && file.labels(i)~=0)
        count = count + 1;
    end;
    if(file.labels(i-1)==0 && file.labels(i)==1)
        startpoint1(i,1)=file.timeVector(i);
        startpoint1(i,2)=i;
    elseif(file.labels(i-1)==0 && file.labels(i)==2)
        startpoint2(i,1)=file.timeVector(i);
        startpoint2(i,2)=i;
    elseif(file.labels(i-1)==0 && file.labels(i)==3)
        startpoint3(i,1)=file.timeVector(i);
        startpoint3(i,2)=i;
    elseif(file.labels(i-1)==0 && file.labels(i)==4)
        startpoint4(i,1)=file.timeVector(i);
        startpoint4(i,2)=i;
    elseif(file.labels(i-1)==0 && file.labels(i)==5)
        startpoint5(i,1)=file.timeVector(i);
        startpoint5(i,2)=i;
    end;
    
end;
startpoint1(any(startpoint1,2)==0,:)=[];%30*2
startpoint2(any(startpoint2,2)==0,:)=[];
startpoint3(any(startpoint3,2)==0,:)=[];
startpoint4(any(startpoint4,2)==0,:)=[];
startpoint5(any(startpoint5,2)==0,:)=[];%40*2

%construct responses
data1 = zeros(batchsize,numCh,30);
data2 = zeros(batchsize,numCh,30);
data3 = zeros(batchsize,numCh,30);
data4 = zeros(batchsize,numCh,30);
data5 = zeros(batchsize,numCh,40);

for x = 1:size(startpoint1,1)
      data1(:,:,x) =  file.signal(startpoint1(x,2):startpoint1(x,2)+batchsize-1,:);
      data2(:,:,x) =  file.signal(startpoint2(x,2):startpoint2(x,2)+batchsize-1,:);
      data3(:,:,x) =  file.signal(startpoint3(x,2):startpoint3(x,2)+batchsize-1,:);
      data4(:,:,x) =  file.signal(startpoint4(x,2):startpoint4(x,2)+batchsize-1,:);
end;


for x = 1:size(startpoint5,1)
      data5(:,:,x) =  file.signal(startpoint5(x,2):startpoint5(x,2)+batchsize-1,:);
end;

[B,A]=butter(5,[8*(2/fs),30*(2/fs)]); %%filter the signal
data1=filter(B,A,data1);
data2=filter(B,A,data2);
data3=filter(B,A,data3);
data4=filter(B,A,data4);
data5=filter(B,A,data5);

trainingData=cat(3,data1,data2,data3,data4);
trainingLabel = cat(1,ones(30,1),2*ones(30,1),3*ones(30,1),4*ones(30,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1.CSP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
nbFilterPairs=1; %%select the number of filter pairs
winSize=[1.5,3];            % time window in seconds [1,2.5][1.5,3][2,3.5][4,6]
winSize=(winSize)*fs;       % time window in samples

%left hand VS right hand, 1 VS 2
labels1=[zeros(size(data1,3),1);ones(size(data2,3),1)];
c1=data1(winSize(1,1):winSize(1,2),:,:);
c2=data2(winSize(1,1):winSize(1,2),:,:);
CSPmatrix1=CSP(covarianceMatrices(c1),covarianceMatrices(c2)); %%generate the CSP matrix
projection1=trialsProjection(cat(3,c1,c2),CSPmatrix1,nbFilterPairs); %%project the EEG signals
features1=extractFeatures(projection1); %%Generate features based on variance
%%%%generate LDA model
classificationModel1=fitcdiscr(features1,labels1); 
K=classificationModel1.Coeffs(1,2).Const;
L=classificationModel1.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(features1(labels1==0,1),features1(labels1==0,2),...
    'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(features1(labels1==1,1),features1(labels1==1,2),...
    'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features1(:,1))) max(max(features1(:,1))) min(min(features1(:,2))) max(max(features1(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 1&2')

%%%prepare for predicting testing data
projection10=trialsProjection(data5(winSize(1,1):winSize(1,2),:,:),CSPmatrix1,nbFilterPairs);
features_test1=extractFeatures(projection10);
[predictedLabels1,scores1]=predict(classificationModel1,features_test1);
%%%prepare for predicting training data
projection100=trialsProjection(trainingData(winSize(1,1):winSize(1,2),:,:),CSPmatrix1,nbFilterPairs);
features_test10=extractFeatures(projection100);
[predictedLabels10,scores10]=predict(classificationModel1,features_test10);

%left hand VS both hands, 1 VS 3
labels2=[zeros(size(data1,3),1);ones(size(data3,3),1)];
c3=data1(winSize(1,1):winSize(1,2),:,:);
c4=data3(winSize(1,1):winSize(1,2),:,:);
CSPmatrix2=CSP(covarianceMatrices(c3),covarianceMatrices(c4)); %%generate the CSP matrix
projection2=trialsProjection(cat(3,c3,c4),CSPmatrix2,nbFilterPairs); %%project the EEG signals
features2=extractFeatures(projection2); %%Generate features based on variance
%%%%generate LDA model
classificationModel2=fitcdiscr(features2,labels2); 
K=classificationModel2.Coeffs(1,2).Const;
L=classificationModel2.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(features2(labels2==0,1),features2(labels2==0,2),...
    'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(features2(labels2==1,1),features2(labels2==1,2),...
    'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features2(:,1))) max(max(features2(:,1))) min(min(features2(:,2))) max(max(features2(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 1&3')

%%%prepare for predicting testing data
projection20=trialsProjection(data5(winSize(1,1):winSize(1,2),:,:),CSPmatrix2,nbFilterPairs);
features_test2=extractFeatures(projection20);
[predictedLabels2,scores2]=predict(classificationModel2,features_test2);
%%%prepare for predicting training data
projection200=trialsProjection(trainingData(winSize(1,1):winSize(1,2),:,:),CSPmatrix2,nbFilterPairs);
features_test20=extractFeatures(projection200);
[predictedLabels20,scores20]=predict(classificationModel2,features_test20);

%left hand VS feet, 1 VS 4
labels3=[zeros(size(data1,3),1);ones(size(data4,3),1)];
c5=data1(winSize(1,1):winSize(1,2),:,:);
c6=data4(winSize(1,1):winSize(1,2),:,:);
CSPmatrix3=CSP(covarianceMatrices(c5),covarianceMatrices(c6)); %%generate the CSP matrix
projection3=trialsProjection(cat(3,c5,c6),CSPmatrix3,nbFilterPairs); %%project the EEG signals
features3=extractFeatures(projection3); %%Generate features based on variance
%%%%generate LDA model
classificationModel3=fitcdiscr(features3,labels3); 
K=classificationModel3.Coeffs(1,2).Const;
L=classificationModel3.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(features3(labels3==0,1),features3(labels3==0,2),...
    'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(features3(labels3==1,1),features3(labels3==1,2),...
    'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features3(:,1))) max(max(features3(:,1))) min(min(features3(:,2))) max(max(features3(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 1&4')

%%%prepare for predicting testing data
projection30=trialsProjection(data5(winSize(1,1):winSize(1,2),:,:),CSPmatrix3,nbFilterPairs);
features_test3=extractFeatures(projection30);
[predictedLabels3,scores3]=predict(classificationModel3,features_test3);
%%%prepare for predicting training data
projection300=trialsProjection(trainingData(winSize(1,1):winSize(1,2),:,:),CSPmatrix3,nbFilterPairs);
features_test30=extractFeatures(projection300);
[predictedLabels30,scores30]=predict(classificationModel3,features_test30);

%right hand VS both hands, 2 VS 3
labels4=[zeros(size(data2,3),1);ones(size(data3,3),1)];
c7=data2(winSize(1,1):winSize(1,2),:,:);
c8=data3(winSize(1,1):winSize(1,2),:,:);
CSPmatrix4=CSP(covarianceMatrices(c7),covarianceMatrices(c8)); %%generate the CSP matrix
projection4=trialsProjection(cat(3,c7,c8),CSPmatrix4,nbFilterPairs); %%project the EEG signals
features4=extractFeatures(projection4); %%Generate features based on variance
%%%%generate LDA model
classificationModel4=fitcdiscr(features4,labels4); 
K=classificationModel4.Coeffs(1,2).Const;
L=classificationModel4.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(features4(labels4==0,1),features4(labels4==0,2),...
    'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(features4(labels4==1,1),features4(labels4==1,2),...
    'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features4(:,1))) max(max(features4(:,1))) min(min(features4(:,2))) max(max(features4(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 2&3')

%%%prepare for predicting testing data
projection40=trialsProjection(data5(winSize(1,1):winSize(1,2),:,:),CSPmatrix4,nbFilterPairs);
features_test4=extractFeatures(projection40);
[predictedLabels4,scores4]=predict(classificationModel4,features_test4);
%%%prepare for predicting training data
projection400=trialsProjection(trainingData(winSize(1,1):winSize(1,2),:,:),CSPmatrix4,nbFilterPairs);
features_test40=extractFeatures(projection400);
[predictedLabels40,scores40]=predict(classificationModel4,features_test40);

%right hand VS feet, 2 VS 4
labels5=[zeros(size(data2,3),1);ones(size(data4,3),1)];
c9=data2(winSize(1,1):winSize(1,2),:,:);
c10=data4(winSize(1,1):winSize(1,2),:,:);
CSPmatrix5=CSP(covarianceMatrices(c9),covarianceMatrices(c10)); %%generate the CSP matrix
projection5=trialsProjection(cat(3,c9,c10),CSPmatrix5,nbFilterPairs); %%project the EEG signals
features5=extractFeatures(projection5); %%Generate features based on variance
%%%%generate LDA model
classificationModel5=fitcdiscr(features5,labels5); 
K=classificationModel5.Coeffs(1,2).Const;
L=classificationModel5.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(features5(labels5==0,1),features5(labels5==0,2),...
    'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(features5(labels5==1,1),features5(labels5==1,2),...
    'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features5(:,1))) max(max(features5(:,1))) min(min(features5(:,2))) max(max(features5(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 2&4')

%%%prepare for predicting testing data
projection50=trialsProjection(data5(winSize(1,1):winSize(1,2),:,:),CSPmatrix5,nbFilterPairs);
features_test5=extractFeatures(projection50);
[predictedLabels5,scores5]=predict(classificationModel5,features_test5);
%%%prepare for predicting training data
projection500=trialsProjection(trainingData(winSize(1,1):winSize(1,2),:,:),CSPmatrix5,nbFilterPairs);
features_test50=extractFeatures(projection500);
[predictedLabels50,scores50]=predict(classificationModel5,features_test50);

%both hands VS feet, 3 VS 4
labels6=[zeros(size(data3,3),1);ones(size(data4,3),1)];
c11=data3(winSize(1,1):winSize(1,2),:,:);
c12=data4(winSize(1,1):winSize(1,2),:,:);
CSPmatrix6=CSP(covarianceMatrices(c11),covarianceMatrices(c12)); %%generate the CSP matrix
projection6=trialsProjection(cat(3,c11,c12),CSPmatrix6,nbFilterPairs); %%project the EEG signals
features6=extractFeatures(projection6); %%Generate features based on variance
%%%%generate LDA model
classificationModel6=fitcdiscr(features6,labels6); 
K=classificationModel6.Coeffs(1,2).Const;
L=classificationModel6.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(features6(labels6==0,1),features6(labels6==0,2),...
    'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(features6(labels6==1,1),features6(labels6==1,2),...
    'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features6(:,1))) max(max(features6(:,1))) min(min(features6(:,2))) max(max(features6(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 3&4')

%%%prepare for predicting testing data
projection60=trialsProjection(data5(winSize(1,1):winSize(1,2),:,:),CSPmatrix6,nbFilterPairs);
features_test6=extractFeatures(projection60);
[predictedLabels6,scores6]=predict(classificationModel6,features_test6);
%%%prepare for predicting training data
projection600=trialsProjection(trainingData(winSize(1,1):winSize(1,2),:,:),CSPmatrix6,nbFilterPairs);
features_test60=extractFeatures(projection600);
[predictedLabels60,scores60]=predict(classificationModel6,features_test60);

%predict labels of all "NaN" trials
p1 = (scores1(:,1)+scores2(:,1)+scores3(:,1))/3;
p2 = (scores1(:,2)+scores4(:,1)+scores5(:,1))/3;
p3 = (scores2(:,2)+scores4(:,2)+scores6(:,1))/3;
p4 = (scores3(:,2)+scores5(:,2)+scores6(:,2))/3;

testingLabels1=zeros(length(p1),1);%labels of all "NaN" trials
[~,testingLabels1]=max([p1 p2 p3 p4],[],2);

%predict labels of trainingData
p10 = (scores10(:,1)+scores20(:,1)+scores30(:,1))/3;
p20 = (scores10(:,2)+scores40(:,1)+scores50(:,1))/3;
p30 = (scores20(:,2)+scores40(:,2)+scores60(:,1))/3;
p40 = (scores30(:,2)+scores50(:,2)+scores60(:,2))/3;

testingLabels10=zeros(length(p10),1);%labels of all known trials
[~,testingLabels10]=max([p10 p20 p30 p40],[],2);
acc = 100*length(find((trainingLabel-testingLabels10)==0))/length(trainingLabel)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2.LDA without CSP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%correlation
frequencyRange=[0 30];
winSize=[2,3.5];            % time window in seconds [1,2.5][1.5,3][2,3.5]
winSize=(winSize)*fs;       % time window in samples


%left hand VS right hand, 1 VS 2
%%compute periodograms
spect_1=abs(fft(data1(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data1,2),size(data1,3)])));                                   
spect_1=spect_1(1:floor(end/2),:,:);
spect_2=abs(fft(data2(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data2,2),size(data2,3)]))); 
spect_2=spect_2(1:floor(end/2),:,:);

allResponses=cat(3,spect_1,spect_2);
labels1=[zeros(size(data1,3),1);ones(size(data2,3),1)];
r2=[];
for ch=1:numCh                                                                      % over all channels
    r2=cat(1,r2,corr(labels1,squeeze(allResponses(:,ch,:))').^2);                      % calculate correlation between the classes
end
frequencyVector=linspace(0,fs/2,size(r2,2))';                              % create frequency vector
[~,frequencyRange]=min(abs(repmat(frequencyRange,length(frequencyVector),1)-repmat(frequencyVector,1,2)));
figure('position',[200,100,900,500],'numbertitle',...
    'off','name','Correlation')
imagesc(flipud(r2(:,frequencyRange(1):frequencyRange(2))))
title('Correlation for class 1&2')
colormap('jet')
set(gca,'xtick',1:+3:diff(frequencyRange)+1,'XTickLabel',...
    num2str(frequencyVector(frequencyRange(1):+3:frequencyRange(2)),'%.1f'))
xlabel('Frequency [Hz]')
set(gca,'ytick',1:26,'YTickLabel',fliplr(channels),'fontweight','bold','fontsize',12)
ylabel('EEG channels')
colorbar

c1=[squeeze(spect_1(16,13,:)) squeeze(spect_1(16,20,:))];
c2=[squeeze(spect_2(16,13,:)) squeeze(spect_2(16,20,:))];

%%Train LDA model
features1=[c1;c2];
classificationModel1=fitcdiscr(features1,labels1); 
K=classificationModel1.Coeffs(1,2).Const;
L=classificationModel1.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(c1(:,1),c1(:,2),'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(c2(:,1),c2(:,2),'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features1(:,1))) max(max(features1(:,1))) min(min(features1(:,2))) max(max(features1(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 1&2')

%%%prepare for predicting testing data
spect_55=abs(fft(data5(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data5,2),size(data5,3)])));                                   
spect_55=spect_55(1:floor(end/2),:,:);
c5c = [squeeze(spect_55(16,13,:)) squeeze(spect_55(16,20,:))];
features_test1=c5c;
[predictedLabels1,scores1]=predict(classificationModel1,features_test1);
%%%prepare for predicting training data
spect_00=abs(fft(trainingData(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(trainingData,2),size(trainingData,3)])));                                   
spect_00=spect_00(1:floor(end/2),:,:);
c0c = [squeeze(spect_00(16,13,:)) squeeze(spect_00(16,20,:))];
features_test10=c0c;
[predictedLabels10,scores10]=predict(classificationModel1,features_test10);

%left hand VS both hands, 1 VS 3
frequencyRange=[0 30];
%%compute periodograms
spect_3=abs(fft(data1(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data1,2),size(data1,3)])));                                   
spect_3=spect_3(1:floor(end/2),:,:);
spect_4=abs(fft(data3(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data3,2),size(data3,3)]))); 
spect_4=spect_4(1:floor(end/2),:,:);

allResponses=cat(3,spect_3,spect_4);
labels2=[zeros(size(data1,3),1);ones(size(data3,3),1)];
r2=[];
for ch=1:numCh                                                                      % over all channels
    r2=cat(1,r2,corr(labels2,squeeze(allResponses(:,ch,:))').^2);                      % calculate correlation between the classes
end
frequencyVector=linspace(0,fs/2,size(r2,2))';                              % create frequency vector
[~,frequencyRange]=min(abs(repmat(frequencyRange,length(frequencyVector),1)-repmat(frequencyVector,1,2)));
figure('position',[200,100,900,500],'numbertitle',...
    'off','name','Correlation')
imagesc(flipud(r2(:,frequencyRange(1):frequencyRange(2))))
title('Correlation for class 1&3')
colormap('jet')
set(gca,'xtick',1:+3:diff(frequencyRange)+1,'XTickLabel',...
    num2str(frequencyVector(frequencyRange(1):+3:frequencyRange(2)),'%.1f'))
xlabel('Frequency [Hz]')
set(gca,'ytick',1:26,'YTickLabel',fliplr(channels),'fontweight','bold','fontsize',12)
ylabel('EEG channels')
colorbar

c3=[squeeze(spect_3(16,13,:)) squeeze(spect_3(16,20,:))];
c4=[squeeze(spect_4(16,13,:)) squeeze(spect_4(16,20,:))];

%%Train LDA model
features2=[c3;c4];
classificationModel2=fitcdiscr(features2,labels2); 
K=classificationModel2.Coeffs(1,2).Const;
L=classificationModel2.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(c3(:,1),c3(:,2),'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(c4(:,1),c4(:,2),'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features2(:,1))) max(max(features2(:,1))) min(min(features2(:,2))) max(max(features2(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 1&3')

%%%prepare for predicting testing data
c5c = [squeeze(spect_55(16,13,:)) squeeze(spect_55(16,20,:))];
features_test2=c5c;
[predictedLabels2,scores2]=predict(classificationModel2,features_test2);
%%%prepare for predicting training data
c0c = [squeeze(spect_00(16,13,:)) squeeze(spect_00(16,20,:))];
features_test20=c0c;
[predictedLabels20,scores20]=predict(classificationModel2,features_test20);

%left hand VS feet, 1 VS 4
frequencyRange=[0 30];
%%compute periodograms
spect_5=abs(fft(data1(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data1,2),size(data1,3)])));                                   
spect_5=spect_5(1:floor(end/2),:,:);
spect_6=abs(fft(data4(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data4,2),size(data4,3)]))); 
spect_6=spect_6(1:floor(end/2),:,:);

allResponses=cat(3,spect_5,spect_6);
labels3=[zeros(size(data1,3),1);ones(size(data4,3),1)];
r2=[];
for ch=1:numCh                                                                      % over all channels
    r2=cat(1,r2,corr(labels3,squeeze(allResponses(:,ch,:))').^2);                      % calculate correlation between the classes
end
frequencyVector=linspace(0,fs/2,size(r2,2))';                              % create frequency vector
[~,frequencyRange]=min(abs(repmat(frequencyRange,length(frequencyVector),1)-repmat(frequencyVector,1,2)));
figure('position',[200,100,900,500],'numbertitle',...
    'off','name','Correlation')
imagesc(flipud(r2(:,frequencyRange(1):frequencyRange(2))))
title('Correlation for class 1&4')
colormap('jet')
set(gca,'xtick',1:+3:diff(frequencyRange)+1,'XTickLabel',...
    num2str(frequencyVector(frequencyRange(1):+3:frequencyRange(2)),'%.1f'))
xlabel('Frequency [Hz]')
set(gca,'ytick',1:26,'YTickLabel',fliplr(channels),'fontweight','bold','fontsize',12)
ylabel('EEG channels')
colorbar

c5=[squeeze(spect_5(16,24,:)) squeeze(spect_5(16,25,:))];
c6=[squeeze(spect_6(16,24,:)) squeeze(spect_6(16,25,:))];

%%Train LDA model
features3=[c5;c6];
classificationModel3=fitcdiscr(features3,labels3); 
K=classificationModel3.Coeffs(1,2).Const;
L=classificationModel3.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(c5(:,1),c5(:,2),'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(c6(:,1),c6(:,2),'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features3(:,1))) max(max(features3(:,1))) min(min(features3(:,2))) max(max(features3(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 1&4')

%%%prepare for predicting testing data
c5c = [squeeze(spect_55(16,24,:)) squeeze(spect_55(16,25,:))];
features_test3=c5c;
[predictedLabels3,scores3]=predict(classificationModel3,features_test3);
%%%prepare for predicting training data
c0c = [squeeze(spect_00(16,24,:)) squeeze(spect_00(16,25,:))];
features_test30=c0c;
[predictedLabels30,scores30]=predict(classificationModel3,features_test30);

%right hand VS both hands, 2 VS 3
frequencyRange=[0 30];
%%compute periodograms
spect_7=abs(fft(data2(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data2,2),size(data2,3)])));                                   
spect_7=spect_7(1:floor(end/2),:,:);
spect_8=abs(fft(data3(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data3,2),size(data3,3)]))); 
spect_8=spect_8(1:floor(end/2),:,:);

allResponses=cat(3,spect_7,spect_8);
labels4=[zeros(size(data2,3),1);ones(size(data3,3),1)];
r2=[];
for ch=1:numCh                                                                      % over all channels
    r2=cat(1,r2,corr(labels4,squeeze(allResponses(:,ch,:))').^2);                      % calculate correlation between the classes
end
frequencyVector=linspace(0,fs/2,size(r2,2))';                              % create frequency vector
[~,frequencyRange]=min(abs(repmat(frequencyRange,length(frequencyVector),1)-repmat(frequencyVector,1,2)));
figure('position',[200,100,900,500],'numbertitle',...
    'off','name','Correlation')
imagesc(flipud(r2(:,frequencyRange(1):frequencyRange(2))))
title('Correlation for class 2&3')
colormap('jet')
set(gca,'xtick',1:+3:diff(frequencyRange)+1,'XTickLabel',...
    num2str(frequencyVector(frequencyRange(1):+3:frequencyRange(2)),'%.1f'))
xlabel('Frequency [Hz]')
set(gca,'ytick',1:26,'YTickLabel',fliplr(channels),'fontweight','bold','fontsize',12)
ylabel('EEG channels')
colorbar

c7=[squeeze(spect_7(16,24,:)) squeeze(spect_7(16,25,:))];
c8=[squeeze(spect_8(16,24,:)) squeeze(spect_8(16,25,:))];

%%Train LDA model
features4=[c7;c8];
classificationModel4=fitcdiscr(features4,labels4); 
K=classificationModel4.Coeffs(1,2).Const;
L=classificationModel4.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(c7(:,1),c7(:,2),'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(c8(:,1),c8(:,2),'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features4(:,1))) max(max(features4(:,1))) min(min(features4(:,2))) max(max(features4(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 2&3')

%%%prepare for predicting testing data
c5c = [squeeze(spect_55(16,24,:)) squeeze(spect_55(16,25,:))];
features_test4=c5c;
[predictedLabels4,scores4]=predict(classificationModel4,features_test4);
%%%prepare for predicting training data
c0c = [squeeze(spect_00(16,24,:)) squeeze(spect_00(16,25,:))];
features_test40=c0c;
[predictedLabels40,scores40]=predict(classificationModel4,features_test40);

%right hand VS feet, 2 VS 4
frequencyRange=[0 30];
%%compute periodograms
spect_9=abs(fft(data2(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data2,2),size(data2,3)])));                                   
spect_9=spect_9(1:floor(end/2),:,:);
spect_10=abs(fft(data4(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data4,2),size(data4,3)]))); 
spect_10=spect_10(1:floor(end/2),:,:);

allResponses=cat(3,spect_9,spect_10);
labels5=[zeros(size(data2,3),1);ones(size(data4,3),1)];
r2=[];
for ch=1:numCh                                                                      % over all channels
    r2=cat(1,r2,corr(labels5,squeeze(allResponses(:,ch,:))').^2);                      % calculate correlation between the classes
end
frequencyVector=linspace(0,fs/2,size(r2,2))';                              % create frequency vector
[~,frequencyRange]=min(abs(repmat(frequencyRange,length(frequencyVector),1)-repmat(frequencyVector,1,2)));
figure('position',[200,100,900,500],'numbertitle',...
    'off','name','Correlation')
imagesc(flipud(r2(:,frequencyRange(1):frequencyRange(2))))
title('Correlation for class 2&4')
colormap('jet')
set(gca,'xtick',1:+3:diff(frequencyRange)+1,'XTickLabel',...
    num2str(frequencyVector(frequencyRange(1):+3:frequencyRange(2)),'%.1f'))
xlabel('Frequency [Hz]')
set(gca,'ytick',1:26,'YTickLabel',fliplr(channels),'fontweight','bold','fontsize',12)
ylabel('EEG channels')
colorbar

c9=[squeeze(spect_9(16,13,:)) squeeze(spect_9(16,20,:))];
c10=[squeeze(spect_10(16,13,:)) squeeze(spect_10(16,20,:))];

%%Train LDA model
features5=[c9;c10];
classificationModel5=fitcdiscr(features5,labels5); 
K=classificationModel5.Coeffs(1,2).Const;
L=classificationModel5.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(c9(:,1),c9(:,2),'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(c10(:,1),c10(:,2),'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features5(:,1))) max(max(features5(:,1))) min(min(features5(:,2))) max(max(features5(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 2&4')

%%%prepare for predicting testing data
c5c = [squeeze(spect_55(16,13,:)) squeeze(spect_55(16,20,:))];
features_test5=c5c;
[predictedLabels5,scores5]=predict(classificationModel5,features_test5);
%%%prepare for predicting training data
c0c = [squeeze(spect_00(16,13,:)) squeeze(spect_00(16,20,:))];
features_test50=c0c;
[predictedLabels50,scores50]=predict(classificationModel5,features_test50);

%both hands VS feet, 3 VS 4
frequencyRange=[0 30];
%%compute periodograms
spect_11=abs(fft(data3(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data3,2),size(data3,3)])));                                   
spect_11=spect_11(1:floor(end/2),:,:);
spect_12=abs(fft(data4(winSize(1,1):winSize(1,2),:,:).*...
    repmat(hamming(diff(winSize)+1),[1,size(data4,2),size(data4,3)]))); 
spect_12=spect_12(1:floor(end/2),:,:);

allResponses=cat(3,spect_11,spect_12);
labels6=[zeros(size(data3,3),1);ones(size(data4,3),1)];
r2=[];
for ch=1:numCh                                                                      % over all channels
    r2=cat(1,r2,corr(labels6,squeeze(allResponses(:,ch,:))').^2);                      % calculate correlation between the classes
end
frequencyVector=linspace(0,fs/2,size(r2,2))';                              % create frequency vector
[~,frequencyRange]=min(abs(repmat(frequencyRange,length(frequencyVector),1)-repmat(frequencyVector,1,2)));
figure('position',[200,100,900,500],'numbertitle',...
    'off','name','Correlation')
imagesc(flipud(r2(:,frequencyRange(1):frequencyRange(2))))
title('Correlation for class 3&4')
colormap('jet')
set(gca,'xtick',1:+3:diff(frequencyRange)+1,'XTickLabel',...
    num2str(frequencyVector(frequencyRange(1):+3:frequencyRange(2)),'%.1f'))
xlabel('Frequency [Hz]')
set(gca,'ytick',1:26,'YTickLabel',fliplr(channels),'fontweight','bold','fontsize',12)
ylabel('EEG channels')
colorbar

c11=[squeeze(spect_11(16,9,:)) squeeze(spect_11(16,25,:))];
c12=[squeeze(spect_12(16,9,:)) squeeze(spect_12(16,25,:))];

%%Train LDA model
features6=[c11;c12];
classificationModel6=fitcdiscr(features6,labels6); 
K=classificationModel6.Coeffs(1,2).Const;
L=classificationModel6.Coeffs(1,2).Linear;
f=@(x1,x2) K + L(1)*x1 + L(2)*x2;

figure
hold on
plot(c11(:,1),c11(:,2),'p','color','w','MarkerFaceColor',color_1,'markerSize',10)
plot(c12(:,1),c12(:,2),'o','color','w','MarkerFaceColor',color_2,'markerSize',8)
model=fimplicit(f,...
    [min(min(features6(:,1))) max(max(features6(:,1))) min(min(features6(:,2))) max(max(features6(:,2)))]);
model.Color=[0 0 0];
model.LineWidth=2;
title('Decision boundary for class 3&4')

%%%prepare for predicting testing data
c5c = [squeeze(spect_55(16,9,:)) squeeze(spect_55(16,25,:))];
features_test6=c5c;
[predictedLabels6,scores6]=predict(classificationModel6,features_test6);
%%%prepare for predicting training data
c0c = [squeeze(spect_00(16,9,:)) squeeze(spect_00(16,25,:))];
features_test60=c0c;
[predictedLabels60,scores60]=predict(classificationModel6,features_test60);

%predict labels of all "NaN" trials
p1 = (scores1(:,1)+scores2(:,1)+scores3(:,1))/3;
p2 = (scores1(:,2)+scores4(:,1)+scores5(:,1))/3;
p3 = (scores2(:,2)+scores4(:,2)+scores6(:,1))/3;
p4 = (scores3(:,2)+scores5(:,2)+scores6(:,2))/3;

testingLabels2=zeros(length(p1),1);%labels of all "NaN" trials
[~,testingLabels2]=max([p1 p2 p3 p4],[],2);

%predict labels of trainingData
p10 = (scores10(:,1)+scores20(:,1)+scores30(:,1))/3;
p20 = (scores10(:,2)+scores40(:,1)+scores50(:,1))/3;
p30 = (scores20(:,2)+scores40(:,2)+scores60(:,1))/3;
p40 = (scores30(:,2)+scores50(:,2)+scores60(:,2))/3;

testingLabels20=zeros(length(p10),1);%labels of all known trials
[~,testingLabels20]=max([p10 p20 p30 p40],[],2);
acc = 100*length(find((trainingLabel-testingLabels20)==0))/length(trainingLabel)
