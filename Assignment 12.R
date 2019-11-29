##Assignment 12 - Practical Machine Learning

setwd() #Masked for Github upload

#Checking for and creating directories
# file.exists("directoryname") - looks to see if the directory exists - T/F
# dir.create("directoryname") - creates a directory if it doesn't exist

#If the directory doesn't exist, make a new one:
if(!file.exists("data")) {
    dir.create("data")
}

## Background ##

#Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. 
#These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, 
#to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, 
#but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
#They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
#More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data ##
#The training data for this project are available here:
    
    #https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

#The test data are available here:
    
    #https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

#The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 
#If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.


## What you should submit ##

#The goal of your project is to predict the manner in which they did the exercise. 
#This is the "classe" variable in the training set. You may use any of the other variables to predict with. 
#You should create a report describing:
    #how you built your model, 
    #how you used cross validation, 
    #what you think the expected out of sample error is, 
    #and why you made the choices you did. 
#The report should be <2000 word with a max of 5 figures.
#You will also use your prediction model to predict 20 different test cases.

#Step 1: Look at the data
    library(caret)
    library(kernlab)
    
    #Using the validation method, I'm going to split the training data into two sub-samples, a training and test set.
    #Then, I'll use the testing data as the validation data on my model.
    
    pml = read.csv("./data/pml-training.csv", header=T)
    
    #Split the data into training, testing, and validation sets. We will use these sub-samples to evaluate the out-of-sample errors later.
        intrain = createDataPartition(y=pml$classe, p=0.7, list=FALSE) #Split 70% of data in training and 30% in testing data
        training = pml[intrain,]
        testing = pml[-intrain,] #Include all samples that don't appear in the "intrain" dataset
        dim(training)
    
        validation = read.csv("./data/pml-testing.csv", header=T)
        dim(validation)
        
    #Remove the variables that contain missing values:
        training = training[, colSums(is.na(training))==0]
        testing = testing[, colSums(is.na(testing))==0]
        validation = validation[, colSums(is.na(validation))==0]
        #Note how this reduces the number of columns!
            dim(training)
            dim(testing)    
            dim(validation)
        
    #What type of outcome is classe? Is it continous or categorical?
        table(training$classe)
            #Categorical!
        
    
    #The first few columns are related to datestamps and timestamps. We don't need these to predict the model.
        #We also don't need the new_window column, or the num_window column. 
        library(dplyr)
        training = training %>% select(-(1:7))
        testing = testing %>% select(-(1:7))
        validation = validation %>% select(-(1:7))
        
        str(training)
        str(training$classe)
        names(training)
        dim(training)
        
    #ALSO! remove the variables that approximate a zero-variance (i.e., these variables are almost fixed and do not vary)    
        #Only do this with the training and test data, NOT the validation data!
        nzv = nearZeroVar(training) #save metrics = shows details
        training = training[, -nzv]
        testing = testing[, -nzv]
        
    #Feature plot - shows how variables are related
        qplot(training$classe)
        featurePlot(x=training[, c("roll_belt", "roll_arm", "roll_dumbbell", "roll_forearm")], 
                    y = training$classe, 
                    plot = "pairs")
        featurePlot(x=training[, c("pitch_belt", "pitch_arm", "pitch_dumbbell", "pitch_forearm")], 
                    y = training$classe, 
                    plot = "pairs")
        featurePlot(x=training[, c("yaw_belt", "yaw_arm", "yaw_dumbbell", "yaw_forearm")], 
                    y = training$classe, 
                    plot = "pairs")
        
        
#Step 2: Preprocessing and standardizing 
    #source: http://rismyhammer.com/ml/Pre-Processing.html
    
    #Investigating correlated predictors
    corrtable = cor(training[, -53])
    summary(corrtable[upper.tri(corrtable)]) #Summary table of the min, 25%, median/mean, 75%, and max correlations
        #Min = -0.99 and max = 0.98!
    
    library(corrplot)
    corrplot(corrtable, order="FPC", method="color", type = "upper", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
    
    library(RANN)
    highcorr = sum(abs(corrtable[upper.tri(corrtable)]) > 0.99)
    
    #Given a correlation matrix, findcorrelation flags predictors for removal:
    highcorrvars = findCorrelation(corrtable, cutoff = 0.75)
    names(training)[highcorrvars]
    dim(training)
    
    #Remove the highly correlated variables from the training set only:
    training = training[, -highcorrvars]
    dim(training)
    
#Step 3: Model building
    #Set seed
    set.seed(12345)
    
    #Build three different models within the same dataset:
    
    #CLASSIFICATION TREE MODEL
        ctmodel = train(classe ~. , method = "rpart", preProcess = c("center", "scale"), data=training)
        library(rattle)
        fancyRpartPlot(ctmodel$finalModel)
    
        #Validate model using the testdata:
        ctpredict = predict(ctmodel, testing)
        cttree = confusionMatrix(ctpredict, testing$classe)    
        ct_accuracy = confusionMatrix(ctpredict, testing$classe)$overall["Accuracy"]
        
        #Plot the accuracy and prediction results:
        plot(cttree$table, col = cttree$byClass, 
             main = "Classification Model Decision Tree", 
             sub = paste("Accuracy = ", round(ct_accuracy, 4)))
        
        
    #RANDOM FOREST MODEL
        #USE PARALLEL PROCESSING TO MAKE THESE RUN FASTER!
        #Source for parallel processing code: https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md
        library(parallel)
        library(doParallel)
        cluster = makeCluster(detectCores() - 1) # convention to leave 1 core for OS
        registerDoParallel(cluster)
        
        #Configure trainControl object 
        #number = # of folds for k-fold cross-validation
        #allowParallel = tells caret to use the cluster that we've registered in the prev step
        fitcontrol = trainControl(method = "cv", 
                                  number = 5, 
                                  allowParallel = TRUE)
        rfmodel = train(classe ~ ., 
                        method = "rf", data=training, 
                        preProcess = c("center", "scale"),
                     trControl = fitcontrol) #Build a random forest model
        
        #De-register parallel processing cluster
        stopCluster(cluster)
        registerDoSEQ() #required! It forces R to return to single-threaded processing
        
        #Validate model using the testdata:
        rfpredict = predict(rfmodel, testing)
        rftree = confusionMatrix(rfpredict, testing$classe)
        rf_accuracy = confusionMatrix(rfpredict, testing$classe)$overall["Accuracy"]
        
        #Plot the accuracy and prediction results:
        plot(rftree$table, col = rftree$byClass, 
             main = "Random Forest Classification Model", 
             sub = paste("Accuracy = ", round(rf_accuracy, 4)))
        
    
    #GENERALIZED BOOSTED REGRESSION MODEL
        #USE PARALLEL PROCESSING TO MAKE THESE RUN FASTER!
        #Source for parallel processing code: https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md
        cluster = makeCluster(detectCores() - 1) # convention to leave 1 core for OS
        registerDoParallel(cluster)
        
        #Configure trainControl object 
        #number = # of folds for k-fold cross-validation
        #allowParallel = tells caret to use the cluster that we've registered in the prev step
        fitcontrol = trainControl(method = "repeatedcv", 
                                  number = 5, 
                                  allowParallel = TRUE, 
                                  repeats = 1)
        boostmodel = train(classe ~ ., method = "gbm", 
                           preProcess = c("center", "scale"),
                           trControl = fitcontrol, data = training)
        #De-register parallel processing cluster
        stopCluster(cluster)
        registerDoSEQ() #required! It forces R to return to single-threaded processing
        
        #Validate model using the testdata:
        boostpredict = predict(boostmodel, testing)
        boosttree = confusionMatrix(boostpredict, testing$classe)
        boost_accuracy = confusionMatrix(boostpredict, testing$classe)$overall["Accuracy"]
        
        #Plot the accuracy and prediction results:
        plot(boosttree$table, col = boosttree$byClass, 
             main = "Random Forest Classification Model", 
             sub = paste("Accuracy = ", round(boost_accuracy, 4)))
        
#Step 4. Ensembler Learning        
        
    #Stack the predictions together using random forests ("rf"). 
        #1. 
        #Build a new dataset that consists of predictions from each model
        #and create a classe variable that is pulled from the classe variable in the testing dataset
        preddf = data.frame(ctpredict, rfpredict, boostpredict, classe=testing$classe)
        
        #2.     
        #Fit a new model that relates the classe variable to the two prediction variables
        #Instead of fitting a model based on the predictors, we are fitting a model based on the predictions!
        combmodfit = train(classe ~ ., method = "rf", 
                           preProcess = c("center", "scale"),
                           data=preddf)
        combpred = predict(combmodfit, preddf)
        
#Step 5. EVALUATE in/out of sample error and accuracy of model!    
        
        #What is the resulting accuracy on the test set? 
        #Is it better or worse than each of the individual predictions? 
        confusionMatrix(ctpredict, testing$classe)$overall["Accuracy"]
        confusionMatrix(rfpredict, testing$classe)$overall["Accuracy"]
        confusionMatrix(boostpredict, testing$classe)$overall["Accuracy"]
        confusionMatrix(combpred, testing$classe)$overall["Accuracy"]
        #Accuracy for the rf model and the combined model are tied!
        
#Step 6. Apply the best model to the validation data
        results = predict(rfmodel, newdata = validation)
        results
        #We can't check these results because each individual's actual class is unknown
        
