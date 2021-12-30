%% Assignment 1A - Bjørnland paper
%%
clc;
clear all; 
close all;

%% Question 1

%% Load data
%note: timeframe in loaded excel files starts from Q1-1982
[xlsdataq, xlstext] = xlsread('Canada Quarterly Data.xls');
vnames = xlstext(1,1:2); % saving columns names
qdata = [ log(xlsdataq(:,2)) xlsdataq(:,3)]; % taking log(GDP) 

[xlsdatam, xlstextm] = xlsread('Canada Monthly Data.xls');
vnamesm = xlstextm(1,1:4); % saving columns names
mdata = xlsdatam(:,2:5);
%% Transformation in quarterly data
%exc rate, foreign rate and domestic rate are transformed
Time = datetime(xlsdatam(:,1),'Convertfrom','excel');
var1= xlsdatam(:,2);%exc rate
var2=xlsdatam(:,3); %foreign rate
var3= xlsdatam(:,5); % domestic rate
TT = timetable(Time,var1,var2,var3);
%move from mothly to quarterly
ExcFedInt = retime(TT, 'quarterly', 'mean');
ExcFedInt = timetable2table(ExcFedInt)
ExcFedInt = table2array(ExcFedInt(:,2:4));
ExcFedInt = [log(ExcFedInt(:,1)) ExcFedInt(:,2:3)]; % take log(exc.rate)
%% Inflation computation from CPI
Infl = 100*(log(xlsdatam(:,4))-lagmatrix(log(xlsdatam(:,4)),12));
TT = timetable(Time,Infl);
Inflq = retime(TT, 'quarterly','mean');
Inflq = timetable2table(Inflq);
Inflq = table2array(Inflq(:,2));
%% Merging and ordering data in a single frame 
% for int - gdp - inflation- dom rate - exc rate (following Bjornland's)
data = [ExcFedInt(:,2) qdata(:,1) Inflq(:,1) qdata(:,2) 100*(ExcFedInt(:,1)-lagmatrix(ExcFedInt(:,1),1))];
%handle date source
Time = datetime(xlsdataq(:,1),'Convertfrom','excel','Format','QQQ-yyyy');
%match time with observations
Time = Time(1:end,1);
data=[datenum(Time) data];
%remove missing observations ( 4 quarters are removed)
data= rmmissing(data);
%redefine time as before
dates=datetime(data(:,1),'Convertfrom','datenum','Format','QQQ-yyyy');
data=data(:,2:end)
%note: After removing missings, timeframe now starts from Q1-1983
%  (same as Bjørnland's paper)
%% Graphs of time series
figure
subplot(2,3,1)
plot(dates,data(:,1),'r','LineWidth',0.8)
title('Foreign interest rate')%foreign interest rate
subplot(2,3,2)
plot(dates,data(:,2),'r','LineWidth',0.8)
title('Log GDP') %gdp
subplot(2,3,3)
plot(dates,data(:,3),'r','LineWidth',0.8)
title('Inflation') %inflation
subplot(2,3,4)
plot(dates,data(:,4),'r','LineWidth',0.8)
title('Domestic interest rate') %interest rate domestic
subplot(2,3,5)
plot(dates,data(:,5),'r','LineWidth',0.8)
title('First differenced exc.rate') %exchange rate change
%% Descriptive statistics
diag = array2table(data, 'VariableNames',{'FedFunds','GDP','Inflation','Intrate','Excrate'});
func=@(x) [mean(x);median(x);max(x);min(x);std(x);acorf(x)];
diagtab= varfun(func,diag);
diagtab.Properties.RowNames={'mean' 'median' 'max' 'min' 'std' 'AutoCorr'};
diagtab.Properties.VariableNames = extractAfter(diagtab.Properties.VariableNames, 'Fun_');
display('Descriptive Statistics')
disp(diagtab)

%% Question 2

%% Determine number of lags for ADF test
% for loop in for loop, table which determines BIC for each model up to
% 13 lags 
array_c= zeros(5,13);
pmax= round(12*(150/100)^(1/4));
for i = 1:5
for q = 1:pmax
 [h,pValue,stat,cValue,reg] = adftest(data(:,i),'model','AR','lags',q);
 array_c(i,q)= reg.BIC;
    
    
end 
end
bic_table= table(array_c(:,1),array_c(:,2),array_c(:,3),array_c(:,4),array_c(:,5),array_c(:,6),array_c(:,7),array_c(:,8),array_c(:,9),array_c(:,10),array_c(:,11),array_c(:,12),array_c(:,13),...
 'VariableNames',{'lag1', 'lag2', 'lag3', 'lag4', 'lag5','lag6', 'lag7','lag8', 'lag9', 'lag10', 'lag11', 'lag12', 'lag13'},...
 'RowNames',{'FedFunds','GDP','Inflation','Intrate','Excrate'})
%With normal model
% for Foreignrate, lag 8       
% for GDP,         lag 1       
% for Inflation,   lag 12      
% for Intrate,     lag 12      
%for Excrate ,     lag 9       


%% ADF test
%Null hypothesis : NON-stationarity of the process
%Foreign rate
[h,pValue,stat,cValue] = adftest(data(:,1),'alpha',0.01,'model',{'AR','ARD','TS'},'lags',8);
array1=[h;pValue;stat;cValue];
bic_table1= table(array1(:,1),array1(:,2),array1(:,3),...
 'VariableNames',{'AD', 'ADR', 'TS'},...
 'RowNames',{'H','pValue','stat','cValue'})
% non-stationary

%Log of GDP
[h,pValue,stat,cValue] = adftest(data(:,2),'alpha',0.01,'model',{'AR','ARD','TS'},'lags',1);
array2=[h;pValue;stat;cValue];
bic_table2= table(array2(:,1),array2(:,2),array2(:,3),...
 'VariableNames',{'AD', 'ADR', 'TS'},...
 'RowNames',{'H','pValue','stat','cValue'}) % always non stationary

%Inflation
[h,pValue,stat,cValue] = adftest(data(:,3),'alpha',0.01,'model',{'AR','ARD','TS'},'lags',12);
array3=[h;pValue;stat;cValue];
bic_table3= table(array3(:,1),array3(:,2),array3(:,3),...
 'VariableNames',{'AD', 'ADR', 'TS'},...
 'RowNames',{'H','pValue','stat','cValue'}) % always non-stationary

%Domestic int rate
[h,pValue,stat,cValue] = adftest(data(:,4),'alpha',0.01,'model',{'AR','ARD','TS'},'lags',12);
array4=[h;pValue;stat;cValue];
bic_table4= table(array4(:,1),array4(:,2),array4(:,3),...
 'VariableNames',{'AD', 'ADR', 'TS'},...
 'RowNames',{'H','pValue','stat','cValue'}) %non-sationary

% First differenced exchange rate
[h,pValue,stat,cValue] = adftest(data(:,5),'alpha',0.01,'model',{'AR','ARD','TS'},'lags',9);
array5=[h;pValue;stat;cValue];
bic_table5= table(array5(:,1),array5(:,2),array5(:,3),...
 'VariableNames',{'AD', 'ADR', 'TS'},...
 'RowNames',{'H','pValue','stat','cValue'}) %stationary 

%% Make time series stationary
%According to Lutkepohl,Kratzig (2004), page 55, we difference the data as
%many times as needed to make it stationary and then repeat the unit root
%test for confirmation
data(:,1)= data(:,1)-lagmatrix(data(:,1),1);
data(:,2)= data(:,2)-lagmatrix(data(:,2),1);
data(:,3)= data(:,3)-lagmatrix(data(:,3),1);
data(:,4)= data(:,4)-lagmatrix(data(:,4),1);
data = data(2:end,:) % drop first row of NaN


%% new graphs of differenced variables
% dates = dates(2:end,:)
% figure
% subplot(2,3,1)
% plot(dates,data(:,1),'r','LineWidth',0.8)
% title('Diff Foreign interest rate')%foreign interest rate
% subplot(2,3,2)
% plot(dates,data(:,2),'r','LineWidth',0.8)
% title('Diff Log GDP') %gdp
% subplot(2,3,3)
% plot(dates,data(:,3),'r','LineWidth',0.8)
% title('Inflation') %inflation
% subplot(2,3,4)
% plot(dates,data(:,4),'r','LineWidth',0.8)
% title('Domestic interest rate') %interest rate domestic
% subplot(2,3,5)
% plot(dates,data(:,5),'r','LineWidth',0.8)
% title('First differenced exc.rate') %exchange rate change
%% Repeat Unit root test with differenced data
[h,pValue,stat,cValue] = adftest(data(:,1),'alpha',0.01,'model',{'AR','ARD','TS'},'lags',8);
array1=[h;pValue;stat;cValue];
bic_table1= table(array1(:,1),array1(:,2),array1(:,3),...
 'VariableNames',{'AD', 'ADR', 'TS'},...
 'RowNames',{'H','pValue','stat','cValue'})
% stationary 

%Log of GDP
[h,pValue,stat,cValue] = adftest(data(:,2),'alpha',0.05,'model',{'AR','ARD','TS'},'lags',1);
array2=[h;pValue;stat;cValue];
bic_table2= table(array2(:,1),array2(:,2),array2(:,3),...
 'VariableNames',{'AD', 'ADR', 'TS'},...
 'RowNames',{'H','pValue','stat','cValue'}) %  stationary

%Inflation
[h,pValue,stat,cValue] = adftest(data(:,3),'alpha',0.01,'model',{'AR','ARD','TS'},'lags',12);
array3=[h;pValue;stat;cValue];
bic_table3= table(array3(:,1),array3(:,2),array3(:,3),...
 'VariableNames',{'AD', 'ADR', 'TS'},...
 'RowNames',{'H','pValue','stat','cValue'}) % stationary

%Domestic int rate
[h,pValue,stat,cValue] = adftest(data(:,4),'alpha',0.01,'model',{'AR','ARD','TS'},'lags',12);
array4=[h;pValue;stat;cValue];
bic_table4= table(array4(:,1),array4(:,2),array4(:,3),...
 'VariableNames',{'AD', 'ADR', 'TS'},...
 'RowNames',{'H','pValue','stat','cValue'}) %stationary

%% Question 3
%code OLS single estimation

[B_hat, SSR, t_ratio, R2, R2_adj, F_stat, Pval] = MyOLS(y,X)



%% Question 4
% Model exchange rate as a function of output, inflation, domestic and
% foreign int.rate allowing for lags

regr= data(:,1:4); %defining the independent variables
T = size(regr,1)
p_max = round(12*(T/100)^(0.25)); % To nearest integer 
Y = data(:,5); %defining the dependent variable
X = [ones(size(regr,1),1)  regr]; %inserting a constant termin the OLS

 [~, results1]= linfo(Y, X) %function that contains MyOLS and information criteria
 
 %Looking at results1, all the information criteria agree on 1 lag length
 
%% Estimation of single OLS
p_choice1= 1
%Estimate OLS one lag
Lags1UR = [X Y]; % need to create lags (including the ones of the dependent variable)
Lags1URl = [X   lagmatrix(Lags1UR(:,2:end),1)]; % Create lags for both independent and dependent variables
Lags1URl = Lags1URl(p_choice1+1:end,:); % Drop nans in independent var

Y_hat = Y(p_choice1+1:end,:); % Drop nans in dependent var

[B_hat1UR, SSR_1UR, t_ratio1, R21, R2_adj1, F_stat1, Pval1] = MyOLS(Y_hat,Lags1URl);

%table representations
%betas and t-values
arrayOLS=[B_hat1UR,t_ratio1];
OLS_table= table(arrayOLS(:,1),arrayOLS(:,2),...
 'VariableNames',{'BETAS', 'T-VALUE'},...
 'RowNames',{'Const','FedFunds','GDP','Inflation','Domestic rate','FedFunds lag1','GDP lag1','Inflation lag1','Domestic rate lag1','Exchange rate lag1'}) %stationary 
%R2, R2adj and fstat + pvalue
arrayVal=[R21, R2_adj1, F_stat1, Pval1];
Val_table= table(arrayVal(:,1),arrayVal(:,2),arrayVal(:,3),arrayVal(:,4),...
 'VariableNames',{'R2', 'R2adj','F-Stat','P-value'})
 



%% Question 5
%% Test for joint significance through F-test
%pvalue computation and significance of coefficients
df1 = size(Lags1URl,2)-1 % number columns in independent matrix minus 1
df2 = T-size(Lags1URl,2) % Length of independent matrix minus number of columns
pvalue = fcdf(F_stat1,df1,df2,"upper")

%reject the null hypothesis, the coefficient are jointly
%significant


%% Question 6

%% Multivariate LS estimator
[Beta,CovBeta,tratioBeta,res,indep,so] = multiLS(y,p,con,tr);
%% Question 7

%% Multivariate information criteria
[AIC, HQC, SIC] = var_info(T, res, K, p)

%% Question 8

%% VAR preparation
forint = data(:,1);
lGDP = data(:,2);
infla = data(:,3);
domint = data(:,4);
excr = data(:,5);
y=[forint lGDP infla domint excr];

[t,K]=size(y); %numeber of observations (t) and number of variables (K)
pmax = round(12*(T/100)^(0.25)); 

%% Find laf length using information criteria
[infomat,sichat,hqchat,aichat]= det_multicriteria(y,pmax); 
% choose a lag of p= 1
p = 2; 
Kp = K*p;
%% Estimate VAR model - With constant and without trend
[Beta,CovBeta,tratioBeta,res,indep,so]=multiLS(y,p,1,0);


%% Question 9-10 
%% Portmanteau test-Autocorrelation in residuals
%all auotcovariances are zero (null hypothesis)
c0 = res'*res/T
c0_1=inv(c0)
h = 5
C= []
for j =1:h
C=[lagmatrix(res,j) res];
C=rmmissing(C)
C=C(:,1:K)'*C(:,K+1:end)/T;
Ph= sum(diag(C'*c0_1*C*c0_1));
adjPh = sum(diag(C'*c0_1*C*c0_1)/(T-j));
end
% adj and non-adj test statistics
Ph = Ph*T %Portmanteau statistic
adjPh = adjPh*T^2  %Ljung-box adjsuted statistics
dfred = K^2*(h-p)
pval_t = chi2cdf (Ph,dfred,'upper')
pval_adj = chi2cdf (adjPh,dfred,'upper')

arrayPort=[Ph, pval_t, adjPh, pval_adj];
Val_table= table(arrayPort(:,1),arrayPort(:,2),arrayPort(:,3),arrayPort(:,4),...
 'VariableNames',{'t-stat','P-value', 'Adj t-stat','adj P-value'})
 
%% Question 11
%% Test for normality
%null hypothesis of normally distributed residuals
[normal]=normtest(res)
pval_norm = chi2cdf (normal,K,'upper') %apparently normality of residuals´assumption is rejected
arrayPort=[Ph, pval_t, adjPh, pval_adj];
Val_table= table(arrayPort(:,1),arrayPort(:,2),arrayPort(:,3),arrayPort(:,4),...
 'VariableNames',{'t-stat','P-value', 'Adj t-stat','adj P-value'})

%% test for ARCH effects
%try it with lags from 1:pmax
lags= 9
[arch_test]=archt(res,lags,K); 
%null hypothesis of no ARCH effects cannot be rejected

%%

    
    


