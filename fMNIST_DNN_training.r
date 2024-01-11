###########run on mnist data set
rm(list=ls())
setwd("/Users/anthonybaptista/Downloads/fMNIST_DNN_training/wk")
library('cccd')
library('dplyr')
library(keras)
library(mlbench)
library(magrittr)
library('pracma')
library(neuralnet)
library("scatterplot3d")
library('igraph')
########################load in data
x_test <- read.csv("fashion-mnist_test.csv")
y_test<-x_test$label
x_test[,-1]->x_test
###
x_train <- read.csv("fashion-mnist_train.csv")
y_train<-x_train$label
x_train[,-1]->x_train
###
rotate <- function(x) t(apply(x, 2, rev))
pdf("fMNIST.pdf",height=5,width = 5)
par(mfrow=c(2,2))
image(rotate(array_reshape(unlist(as.vector(x_train[which(y_train==5)[1],])), c(28,28))),col=grey(seq(0,1,length=180)),axes=FALSE)#sandle
image(rotate(array_reshape(unlist(as.vector(x_train[which(y_train==9)[1],])), c(28,28))),col=grey(seq(0,1,length=180)),axes=FALSE)#ankle boot
dev.off()
#######
#######
table(y_train)
table(y_test)
########maybe we just restict to 5s and 9s as these are ankle boots and sandles
##1 is trousers
c(which(y_train==5),which(y_train==9))->train_1_7
c(which(y_test==5),which(y_test==9))->test_1_7
y_test[test_1_7]->y_test
y_train[train_1_7]->y_train
y_test[which(y_test==5)]<-0;y_test[which(y_test==9)]<-1
y_train[which(y_train==5)]<-0;y_train[which(y_train==9)]<-1
x_train[train_1_7,]->x_train
x_test[test_1_7,]->x_test
dim(x_train)
dim(x_test)
####
as.matrix(x_test)->x_test
as.matrix(x_train)->x_train
########################first build models and output activations at each layer (the computationally demanding step)
b<-5###number of models to build
####outputs=accuracy and layer outputs
accuracy<-list()
activation_list<-list()
for(j in 1:b){
  #####define DNN architectures
  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 50, activation = 'relu', input_shape = c(length(x_test[1,]))) %>%
    layer_dense(units = 50, activation = 'relu') %>%
    layer_dense(units = 50, activation = 'relu') %>%
    layer_dense(units = 50, activation = 'relu') %>%
    layer_dense(units = 50, activation = 'relu') %>%
    layer_dense(units = 1,activation = 'sigmoid')
  #####binary cross-entropy loss function for all models
  summary(model)
  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy'))
  #####train model on training data defined above
  dnn_history <- model %>% fit(x_train,y_train,
                               epochs = 50, batch_size = 32,
                               validation_split = 0.2)
  #####check accuracy on test data - should output this for each model
  model %>% evaluate(x_test, y_test)->accuracy[[j]]###this is critical, needs to be >99% accuracy
  #####output the layers on implementation over test data
  layer_outputs <- lapply(model$layers, function(layer) layer$output)
  activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
  activation_list[[j]] <- activation_model %>% predict(x_test)
}
###############
save(activation_list,accuracy,x_test,y_test,b,file="fmnist_shoes_width_50_depth_5_model_activations.rd")