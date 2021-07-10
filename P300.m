close all
clear
clc

numCh=8;
numTargets=12;
numSeries = 10;
electrodes={...
    'Fz','Cz','P3','Pz','P4','PO7','PO8','Oz'}; %EEG electrodes
donchin=[...
    'A' 'B' 'C' 'D' 'E' 'F';...
    'G' 'H' 'I' 'J' 'K' 'L';...
    'M' 'N' 'O' 'P' 'Q' 'R';...
    'S' 'T' 'U' 'V' 'W' 'X';...
    'Y' 'Z' '1' '2' '3' '4';...
    '5' '6' '7' '8' '9' '_'];  %%Donchin matrix


file = importdata('P300data.mat');
trainingData=file.training;
testingData=file.testing;
startpoint = zeros(size(trainingData,1),3);
startpoint1 = zeros(size(trainingData,1),3);
startpoint2 = zeros(size(trainingData,1),3);
j = 1;
k = 1;
l = 1;
    for i = 1:size(trainingData,1)
      if (trainingData(i,10) ~= 0)
         startpoint(j,1) = trainingData(i,9);%time point
         startpoint(j,2) = i;%index
         startpoint(j,3) = trainingData(i,10);%label
         j = j+1;
      end;
      if (trainingData(i,10) ~= 0 && trainingData(i,11) ~= 0)
         startpoint1(k) = trainingData(i,9);
         startpoint1(k,2) = i;
         startpoint1(k,3) = trainingData(i,10);
         k = k+1;
      elseif (trainingData(i,10) ~= 0 && trainingData(i,11) == 0)
         startpoint2(l) = trainingData(i,9);
         startpoint2(l,2) = i;
         startpoint2(l,3) = trainingData(i,10);
         l = l+1;
      end;
    end;


startpoint(any(startpoint,2)==0,:)=[];%all
startpoint1(any(startpoint1,2)==0,:)=[];%infreq
startpoint2(any(startpoint2,2)==0,:)=[];%freq

%construct infrequent responses
batchsize = 200;%0.8s
infreq = zeros(batchsize,numCh,size(startpoint1,1)/numSeries,numSeries);%200*8*50*10 20
freq = zeros(batchsize,numCh,size(startpoint2,1)/numSeries,numSeries);%200*8*250*10 100
j = 0;
k = 0;
for i=1:size(startpoint1,1)
    round = ceil(i/20);%1-25
    if(startpoint1(i,3) == startpoint1(20*(round-1)+1,3))
       j = j+1;
       infreq(:,:,round*2-1,j) = trainingData(startpoint1(i,2):startpoint1(i,2)+batchsize-1,1:8);
       if(j==10) 
           j=0; 
       end; 
    elseif(startpoint1(i,3) == startpoint1(20*(round-1)+2,3))
       k = k+1;
       infreq(:,:,round*2,k) = trainingData(startpoint1(i,2):startpoint1(i,2)+batchsize-1,1:8);
       if(k==10) 
           k=0; 
       end; 
    end;
end;
infreq = mean(infreq,4);%200*8*50
%construct frequent responses
j1 = 0;
j2 = 0;
j3 = 0;
j4 = 0;
j5 = 0;
j6 = 0;
j7 = 0;
j8 = 0;
j9 = 0;
j10 = 0;
for i=1:size(startpoint2,1)
    round = ceil(i/100);%1-25
    if(startpoint2(i,3) == startpoint2(100*(round-1)+1,3))
       j1 = j1+1;
       freq(:,:,round*10-9,j1) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j1==10) 
           j1=0; 
       end; 
    elseif(startpoint2(i,3) == startpoint2(100*(round-1)+2,3))
       j2 = j2+1;
       freq(:,:,round*10-8,j2) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j2==10) 
           j2=0; 
       end; 
    elseif(startpoint2(i,3) == startpoint2(100*(round-1)+3,3))
       j3 = j3+1;
       freq(:,:,round*10-7,j3) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j3==10) 
           j3=0; 
       end; 
    elseif(startpoint2(i,3) == startpoint2(100*(round-1)+4,3))
       j4 = j4+1;
       freq(:,:,round*10-6,j4) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j4==10) 
           j4=0; 
       end;
    elseif(startpoint2(i,3) == startpoint2(100*(round-1)+5,3))
       j5 = j5+1;
       freq(:,:,round*10-5,j5) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j5==10) 
           j5=0; 
       end; 
    elseif(startpoint2(i,3) == startpoint2(100*(round-1)+6,3))
       j6 = j6+1;
       freq(:,:,round*10-4,j6) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j6==10) 
           j6=0; 
       end; 
    elseif(startpoint2(i,3) == startpoint2(100*(round-1)+7,3))
       j7 = j7+1;
       freq(:,:,round*10-3,j7) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j7==10) 
           j7=0; 
       end; 
    elseif(startpoint2(i,3) == startpoint2(100*(round-1)+8,3))
       j8 = j8+1;
       freq(:,:,round*10-2,j8) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j8==10) 
           j8=0; 
       end; 
    elseif(startpoint2(i,3) == startpoint2(100*(round-1)+9,3))
       j9 = j9+1;
       freq(:,:,round*10-1,j9) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j9==10) 
           j9=0; 
       end; 
     elseif(startpoint2(i,3) == startpoint2(100*(round-1)+10,3))
       j10 = j10+1;
       freq(:,:,round*10,j10) = trainingData(startpoint2(i,2):startpoint2(i,2)+batchsize-1,1:8);
       if(j10==10) 
           j10=0; 
       end; 
    end;
end;
freq = mean(freq,4);%200*8*250


%mean calculation 200*8
infreq0 = mean(infreq,3);
freq0 = mean(freq,3);
figure
for i=1:numCh
    subplot(4,2,i);
    x = 1:batchsize;
    plot(x,freq0(:,i));%channel i
    hold on,
    plot(x,infreq0(:,i),'red');
    xticks(0:205/8:205);
    xticklabels({ '0','100','200','300','400','500','600','700','800' });%ms
    xlabel('ms');
    ylabel('amplitude');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%1.regression%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%regression-train model
labels=[ones(size(infreq,3),1);-ones(size(freq,3),1)];%3000*1
features=reshape(cat(3,infreq,freq),size(infreq,1)*size(infreq,2),length(labels))';
[coefficients,~,PVAL,in,stats]=stepwisefit(features,labels,'maxiter',60,'display','off','penter',0.1,'premove',0.15);
%coefficient estimate, STD, pvalue,in=finalmodel,stats=statistics
[~,indxs]=sort(PVAL(in~=0),'ascend');
coefficients=coefficients(in~=0);
coefficients=coefficients(indxs);
b0=stats.intercept;

features=features(:,in~=0);
features=features(:,indxs);

figure
hold on
plot(features(labels==-1,1),features(labels==-1,2),'o','MarkerEdgeColor',[1 1 1],'MarkerFaceColor','b')
plot(features(labels==1,1),features(labels==1,2),'o','MarkerEdgeColor',[1 1 1],'MarkerFaceColor','r')

%line plot
hold on,
x=linspace(min(features(:,1)),max(features(:,1)),1000);
y=-x*(coefficients(1)/coefficients(2))-(b0/coefficients(2));
plot(x,y,'k','LineWidth',1);
title('regression model based on training set')

distances=(sum(features'.*coefficients)+b0)';
distances(distances>=0)=1;
distances(distances<0)=-1;
acc = 100*length(find((labels-distances)==0))/length(labels)

%regression-test on trainingData
spelledWord=[];
wordlength = 25;
 for sequence=1:wordlength

    responses=zeros(batchsize,numCh,numTargets,numSeries);
    ntest = size(startpoint,1)/wordlength;%120
    
     for x = 1:size(startpoint,1)
       
          if(x <= sequence*ntest && x > (sequence-1)*ntest )
              series = ceil((x-(sequence-1)*ntest)/12);
              responses(:,:,startpoint(x,3),series) = trainingData(startpoint(x,2):startpoint(x,2)+batchsize-1,1:8); 
          end
          
     end
     
     responses=reshape(mean(responses,4),size(responses,1)*size(responses,2),numTargets);
     responses=responses(in~=0,:)';
     features=responses(:,indxs);
     
     [~,row]=max(sum(features(1:6,:).*repmat(coefficients',numTargets/2,1),2)+b0);
     [~,col]=max(sum(features(7:12,:).*repmat(coefficients',numTargets/2,1),2)+b0);

     spelledWord = cat(2,spelledWord,donchin(row,col));       
 end
 spelledWord
 
%regression-test on testingData
startpoint3 = zeros(size(testingData,1),3);
j = 1;
    for i = 1:size(testingData,1)
      if (testingData(i,10) ~= 0)
         startpoint3(j,1) = testingData(i,9);%time point
         startpoint3(j,2) = i;%index
         startpoint3(j,3) = testingData(i,10);%label
         j = j+1;
      end;
    end;

startpoint3(any(startpoint3,2)==0,:)=[];%960 stimulus, 8 letters

spelledWord=[];
wordlength = size(startpoint3,1)/120;
 for sequence=1:wordlength

    responses=zeros(batchsize,numCh,numTargets,numSeries);
    ntest = size(startpoint3,1)/wordlength;%120
    
     for x = 1:size(startpoint3,1)
       
          if(x <= sequence*ntest && x > (sequence-1)*ntest )
              series = ceil((x-(sequence-1)*ntest)/12);
              responses(:,:,startpoint3(x,3),series) = testingData(startpoint3(x,2):startpoint3(x,2)+batchsize-1,1:8); 
          end
          
     end
     
     responses=reshape(mean(responses,4),size(responses,1)*size(responses,2),numTargets);
     responses=responses(in~=0,:)';
     features=responses(:,indxs);
     
     [~,row]=max(sum(features(1:6,:).*repmat(coefficients',numTargets/2,1),2)+b0);
     [~,col]=max(sum(features(7:12,:).*repmat(coefficients',numTargets/2,1),2)+b0);

     spelledWord = cat(2,spelledWord,donchin(row,col));      
 end
 spelledWord
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%2.Fischer discriminant analysis%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Fischer discriminant analysis-train model
labels=[ones(size(infreq,3),1);-ones(size(freq,3),1)];%3000*1
features=reshape(cat(3,infreq,freq),size(infreq,1)*size(infreq,2),length(labels))';
classificationModel=fitcdiscr(features,labels); %generate classification model
[predictedLabels0,scores0]=predict(classificationModel,features); %apply classification model
acc = 100*length(find((labels-predictedLabels0)==0))/length(labels)

K=classificationModel.Coeffs(1,2).Const; %%bias
L=classificationModel.Coeffs(1,2).Linear; %%coefficients

f=@(x1,x2) K + L(1)*x1 + L(2)*x2; %%generate and plot decision boundary
h2=fimplicit(f,[min(min(features(:,1))) max(max(features(:,1)))...
            min(min(features(:,2))) max(max(features(:,2)))]);
h2.Color=[0 0 0]; %%color and length of the decision boundary
h2.LineWidth=2;
title([]);

%Fischer discriminant analysis-test on training data
wordlength = 25;
spelledWord = [];
 for sequence=1:wordlength

    responses=zeros(batchsize,numCh,numTargets,numSeries);
    ntest = size(startpoint,1)/wordlength;%120
    
     for x = 1:size(startpoint,1)
       
          if(x <= sequence*ntest && x > (sequence-1)*ntest )
              series = ceil((x-(sequence-1)*ntest)/12);
              responses(:,:,startpoint(x,3),series) = trainingData(startpoint(x,2):startpoint(x,2)+batchsize-1,1:8); 
          end
          
     end
     
     responses=reshape(mean(responses,4),size(responses,1)*size(responses,2),numTargets)';    
     [predictedLabels,scores]=predict(classificationModel,responses); %%apply classification model
     row = 0;
     col = 0;
     for x = 1:size(predictedLabels,1)
         if(predictedLabels(x,1) == 1&&row==0)
             row = x;
         elseif(predictedLabels(x,1) == 1&&row~=0)
             col = x-6;
         end;
     end
     if(row~=0&&col~=0)
      spelledWord = cat(2,spelledWord,donchin(row,col));    
     else
        spelledWord = cat(2,spelledWord,'*'); %no effective decoded letter
     end;   
 end
 spelledWord
 
%Fischer discriminant analysis-test on testing data
spelledWord=[];
wordlength = size(startpoint3,1)/120;
 for sequence=1:wordlength

    responses=zeros(batchsize,numCh,numTargets,numSeries);
    ntest = size(startpoint3,1)/wordlength;%120
    
     for x = 1:size(startpoint3,1)
       
          if(x <= sequence*ntest && x > (sequence-1)*ntest )
              series = ceil((x-(sequence-1)*ntest)/12);
              responses(:,:,startpoint3(x,3),series) = testingData(startpoint3(x,2):startpoint3(x,2)+batchsize-1,1:8); 
          end
          
     end
     
     responses=reshape(mean(responses,4),size(responses,1)*size(responses,2),numTargets)';    
     [predictedLabels,scores]=predict(classificationModel,responses); %%apply classification model
     row = 0;
     col = 0;
     for x = 1:size(predictedLabels,1)
         if(predictedLabels(x,1) == 1&&row==0)
             row = x;
         elseif(predictedLabels(x,1) == 1&&row~=0)
             col = x-6;
         end;
     end
     if(row~=0&&col~=0)
      spelledWord = cat(2,spelledWord,donchin(row,col));    
     else
      spelledWord = cat(2,spelledWord,'*'); %no effective decoded letter
     end;
 end
spelledWord

