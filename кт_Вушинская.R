setwd("C:/Users/home/Desktop/������ ������/��������� ���� � ����� R")
#������ ������������ ����� ���������� ������� 699 ��������� 
#������ - ���������� ����� �������: ����������������� ��� ���������������
df <-  read.csv('biopsy.csv', sep=',', header = TRUE)
df <- df[3:12]
df <- df [, -6]#���������� ��������� �� ������������, ��� ��� ���� �������� � ������
#����������� ����� � �������� ����������
df$class <- gsub('benign', 0, df$class)
df$class <- gsub('malignant', 1, df$class)
df$class <- as.numeric(df$class)
set.seed(20)

#neuralnet
install.packages('neuralnet')
library('neuralnet')
#������� ��������� �������������� max_min
max_data <- apply(df, 2, max)
min_data <- apply(df, 2, min)
data_scaled <- data.frame(scale(df, center = min_data, scale = max_data - min_data))
#�������� ������ � ����������� ���������� data_scaled
#������� ����� ����������� ��� ���������� ����
index <- sample(1:nrow(df), round(0.80*nrow(df)))
index
#�������� ������� � ������ ��� ���������� ����
#� �������� ������� ��� ������������ ���� �� ���������� ���������� �������
#��� ��������, ���������� ������ ������� ������������� ������� index
#������� � ������������� �������, ��������� �������� ������� � �������� �������
train_data <- as.data.frame(data_scaled[index,])
test_data <- as.data.frame(data_scaled[-index,])
#������� ������� ��� ���� ����������� ���������� ������ �� ������ ��������
n <- names(df)
f <- as.formula(paste('class ~', paste(n[!n %in% 'class'], collapse = '+')))
#�������� ��������� ���� �� ������������� ������, � ������� 2 ������� ���� (5 � 3)
d_net <- neuralnet(f, data = train_data, hidden = c(5, 3), linear.output = F)
plot(d_net)
#��������� �� ���������� �� �������� �������
predicted <- compute(d_net, test_data)
print(head(predicted$net.result))
#�������� ����������
predicted$net.result <- sapply(predicted$net.result, round, digits = 0)
#����� ���������� �������� ������� ��������� �����������
test1 <- table(test_data$class, predicted$net.result)
test1
#�� ������� ��������� ������ ����������, �� �������� - ���������
#��������� �� ���������� 0 � 1 � �������� �������, ����� ��������� ������������
table(test_data$class)
#������ �������� ������
Accuracy_nn <- (test1[1,1] + test1[2, 2])/sum(test1)
Accuracy_nn <- round(Accuracy_nn, digits = 2)
Accuracy_nn

#kohonen
install.packages("kohonen")
library('kohonen')
#������� ������� ������� � ��������� ����������
df2 <- df[1:8]
df2_a <- df[9]
#����� ������� ��������� ������� - 559 ����� (80%), 
#��������� 140 ����� �������������� � �������� �������� �������
train <- sample(nrow(df2), 559)
X_train <- scale(df2[train,])
X_test <- scale(df2[-train,],
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:center"))
train_data2 <- list(measurements = X_train,
                    df2_a = df2_a[train,])
test_data2 <- list(measurements = X_test,
                   df2_a = df2_a[-train,])
#������ ������
mygrid <- somgrid(5, 5, 'hexagonal')
som.df2 <- supersom(train_data2, grid = mygrid) 
som.predict <- predict(som.df2, newdata = test_data2)
#Confusion matrix
test2 <- table(df2_a[-train,], som.predict$predictions$df2_a)
test2
#������ �������� ������
Accuracy_k <- (test2[1,1] + test2[2, 2])/sum(test2)
Accuracy_k <- round(Accuracy_k, digits = 2)
Accuracy_k

#rsns
install.packages('RSNNS')
library(RSNNS)
df3 <- df[sample(1:nrow(df), length(1:nrow(df))), 1:ncol(df)]
df3_Values <- df3[, 1:8]
df3_Target <- df3[, 9]
#�������� �� ������������� � �������� ������� � ����������� ������
df3 <- splitForTrainingAndTest(df3_Values, df3_Target, ratio = 0.2)
df3 <- normTrainingAndTestSet(df3)
#������ ������
model <- mlp(df3$inputsTrain,
             df3$targetsTrain,
             size = 5,
             maxit = 50,
             inputsTest = df3$inputsTest,
             targetsTest = df3$targetsTest)
predictions <- predict(model, df3$inputsTest)
predictions <- sapply(predictions, round, digits = 0)
#Confusion matrix
test3 <- confusionMatrix(df3$targetsTest, predictions)
test3
#������ �������� ������
Accuracy_rsns <- (test3[1,1] + test3[2, 2])/sum(test3)
Accuracy_rsns <- round(Accuracy_rsns, digits = 2)
Accuracy_rsns

#������� ������� �����������
results <- cbind(Accuracy_nn, Accuracy_k, Accuracy_rsns)
rownames(results) <- c("Accuracy")
colnames(results) <- c("NeuralNet", "Kohonen", "RSNNS")
results
#����� ���������� ����������� kohonen, �������� ��������� 100%
#���������� NeuralNet � RSNNS �������� ���������� ��������� - 93%
