rm(list=ls())
setwd("~/Turing Projects/DNNs_curvature_and_chaos/fMNIST/")
c("fmnist_shoes_width_25_depth_5_model_activations.rd")->load_files
library('cccd')
library('dplyr')
library(keras)
library(mlbench)
library(magrittr)
library('pracma')
library(neuralnet)
library("scatterplot3d")
library('igraph')
######for unity weights as we have F_Ricci(i,j)=4-deg(i)-deg(j)
#accuracy
#ll<-1
for(ll in 1:length(load_files)){
  load(load_files[ll])
  k_list<-c(30,120,300)
  a<-0.98
  ###setwd for plots
  dir.create(strsplit(load_files[ll],split = "[.]")[[1]][1])
  setwd(paste0(strsplit(load_files[ll],split = "[.]")[[1]][1],"/"))
  #####set outputs for k loop
  gdists_k<-list()
  FR_k<-list()
  correlation_k<-list()
  correlation_k_shift<-list()
  correlation_k_int<-list()
  n_mods<-list()
  #######loop through k
  start_time <- Sys.time()
  for(ii in 1:length(k_list)){
    print(ii)
    ####set k
    k<-k_list[ii]
    #k<-10
    ####set intermediate list to loop through the models
    scalar_curvs2<-list()
    FR1<-list()
    ####loop through models
    for(j in 1:b){
      print(j)
      ####set activations of model
      activation_list[[j]]->activations
      #####compute kNN graphs
      gs1<-list()
      for(i in 1:(length(activations)-1)){
        av<-activations[[i]]
        nng(av,method = "Euclidean",k=k)->gs1[[i]]
      }
      nng(x_test,method = "Euclidean",k=k)->g0.1
      ####compute FR curvatures for each activations
      F_n_list1<-list()
      for(i in 1:length(gs1)){
        degree(gs1[[i]])->D
        4-outer(D,D,FUN = "+")->F_mat
        a_mat<-as_adjacency_matrix(gs1[[i]],sparse = F,type="both")
        F_mat[which(a_mat==0)]<-0
        apply(F_mat,1,sum)->F_n1
        F_n1->F_n_list1[[i]]
      }
      ################
      Ric1<-list()
      Ric1[[1]]<- (distances(gs1[[1]])-distances(g0.1))
      for(i in 2:length(gs1)){
        Ric1[[i]]<- (distances(gs1[[i]])-distances(gs1[[i-1]]))}
      ###########compute sum of these over data points (consider changing to local sum)
      sc2<-list()
      for(i in 1:length(Ric1)){sc2[[i]]<-apply(Ric1[[i]],1,function(x){return(sum(x,na.rm = T))})}
      ####output 
      sc2->scalar_curvs2[[j]]
      F_n_list1->FR1[[j]]
    }
    scalar_curvs2->scalar_curvs1
    #########consider only models which achieve accuracy above threshold
    as.vector(unlist(accuracy))[c(1:length(accuracy))*2]->acc
    #########generate geodesic data frame
    scalar_curvs1[[which(acc>a)[1]]]->s
    sr<-s[[1]]
    for(i in 2:length(s)){sr<-cbind(sr,s[[i]])}
    #####
    for(i in which(acc>a)[-1]){
      s<-scalar_curvs1[[i]]
      sr2<-s[[1]]
      for(j in 2:length(s)){sr2<-cbind(sr2,s[[j]])}
      cbind(sr,sr2)->sr
    }
    colnames(sr)<-rep(1:length(scalar_curvs1[[1]]),length(which(acc>a)))
    ########extract to data frame summary
    sr[,1]->ssr
    for(i in 2:ncol(sr)){c(ssr,sr[,i])->ssr}
    rep(1,nrow(x_test))->layer
    for(i in 2:length(scalar_curvs1[[1]])){c(layer,rep(i,nrow(x_test)))->layer}
    rep(layer,length(which(acc>a)))->layer_all
    rep(1,length(scalar_curvs1[[1]])*nrow(x_test))->mod
    for(i in 2:length(which(acc>a))){c(mod,rep(i,length(scalar_curvs1[[1]])*nrow(x_test)))->mod}
    data.frame(ssr,layer_all,mod)->sc_data
    names(sc_data)<-c("ssr","layer","mod")
    #######################################
    ###generate FR curvature data frame
    FR1->scalar_curvs1
    as.vector(unlist(accuracy))[c(1:length(accuracy))*2]->acc
    scalar_curvs1[[which(acc>a)[1]]]->s
    sr<-s[[1]]
    for(i in 2:length(s)){sr<-cbind(sr,s[[i]])}
    for(i in which(acc>a)[-1]){
      s<-scalar_curvs1[[i]]
      sr2<-s[[1]]
      for(j in 2:length(s)){sr2<-cbind(sr2,s[[j]])}
      cbind(sr,sr2)->sr
    }
    colnames(sr)<-rep(1:length(scalar_curvs1[[1]]),length(which(acc>a)))
    ########extract to data frame summary
    sr[,1]->ssr
    for(i in 2:ncol(sr)){c(ssr,sr[,i])->ssr}
    rep(1,nrow(x_test))->layer
    for(i in 2:length(scalar_curvs1[[1]])){c(layer,rep(i,nrow(x_test)))->layer}
    rep(layer,length(which(acc>a)))->layer_all
    rep(1,length(scalar_curvs1[[1]])*nrow(x_test))->mod
    for(i in 2:length(which(acc>a))){c(mod,rep(i,length(scalar_curvs1[[1]])*nrow(x_test)))->mod}
    data.frame(ssr,layer_all,mod)->fr_data
    names(fr_data)<-c("ssr","layer","mod")
    ###deal with infinte geodesics by removing models with these (consider there will be a better way involving signulrities)
    aggregate(ssr ~ layer+mod, sc_data, sd)->d
    cbind(d,aggregate(ssr ~ layer+mod, sc_data, mean)[,3])->d
    data.frame(d)->d
    names(d)<-c("layer","mod","sd","mean")
    max(d[-which(d==Inf)],na.rm = T)
    d[c(which(d$mean=="-Inf"),which(d$mean=="Inf")),]$mod->rmod
    if(length(rmod)>0){match(sc_data$mod,rmod)->m
      which(is.na(m)==F)->rm2
      sc_data2<-sc_data[-rm2,]} else {sc_data2<-sc_data}
    #
    if(length(rmod)>0){match(fr_data$mod,rmod)->m
      which(is.na(m)==F)->rm2
      fr_data2<-fr_data[-rm2,]} else {fr_data2<-fr_data}
    #########aggregate the data frames and sum over data points to get a total for each model and layer
    aggregate(ssr ~ layer+mod, sc_data2, sum)->msc
    aggregate(ssr ~ layer+mod, fr_data2, sum)->mfr
    #########consider 2 types of correlation
    cor.test(msc$ssr,mfr$ssr)->aa
    cor.test(msc$ssr[-which(msc$layer==1)],mfr$ssr[-which(mfr$layer==max(mfr$layer))])->aa2
    ############
    gdists_k[[ii]]<-msc
    FR_k[[ii]]<-mfr
    correlation_k[[ii]]<-c(aa$estimate,aa$p.value)
    correlation_k_shift[[ii]]<-c(aa2$estimate,aa2$p.value)
    n_mods[[ii]]<-length(which(acc>a))
    ###############output some relevant figs
    ####################################################
    #if(plot_out=="yes"){
    pdf(file=paste0("k_",k_list[ii],".pdf"),height=8,width = 8)
    par(mfrow=c(2,2))
    ####geodesics over layer
    boxplot(msc$ssr~msc$layer,xlab="layer",ylab="Total geodesic change from prior layer")
    ####FR curvature over layer
    boxplot(mfr$ssr~mfr$layer,xlab="layer",ylab="Total FR Curvature")
    ####skip layer FR curvature vs geodesic
    plot(msc$ssr[-which(msc$layer==1)],mfr$ssr[-which(mfr$layer==max(mfr$layer))],
         col=mfr$layer,pch=16,xlab="Total geodesic change from prior layer from l-1->l",ylab="Total FR Curvature of l-1",main="layer skip")
    abline(lm(mfr$ssr[-which(mfr$layer==max(mfr$layer))]~msc$ssr[-which(msc$layer==1)]))
    dev.off()#}else{NULL}
  }
  end_time <- Sys.time()
  end_time-start_time##15 mins for 6
  ############
  save(paste0("k_all_",load_files[ll]))
}
