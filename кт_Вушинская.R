setwd("C:/Users/home/Desktop/Анализ данных/НЕЙРОННЫЕ СЕТИ В СРЕДЕ R")
#Данные представляют собой результаты биопсии 699 пациентов 
#Задача - определить класс опухоли: доброкачественная или злокачественная
df <-  read.csv('biopsy.csv', sep=',', header = TRUE)
df <- df[3:12]
df <- df [, -6]#переменная исключена из рассмотрения, так как есть пропуски в данных
#Преобразуем класс в бинарную переменную
df$class <- gsub('benign', 0, df$class)
df$class <- gsub('malignant', 1, df$class)
df$class <- as.numeric(df$class)
set.seed(20)

#neuralnet
install.packages('neuralnet')
library('neuralnet')
#Проведём процедуру стандартизации max_min
max_data <- apply(df, 2, max)
min_data <- apply(df, 2, min)
data_scaled <- data.frame(scale(df, center = min_data, scale = max_data - min_data))
#создадим вектор с порядковыми значениями data_scaled
#который будет использован для тренировки сети
index <- sample(1:nrow(df), round(0.80*nrow(df)))
index
#Создадим таблицу с даными для тренировки сети
#и создадим таблицу для тестирования сети из оставшихся порядковых номеров
#все значения, порядковые номера которых соответствуют вектору index
#попадут в тренировочную выборку, остальные значения попадут в тестовую выборку
train_data <- as.data.frame(data_scaled[index,])
test_data <- as.data.frame(data_scaled[-index,])
#зададим функцию для сети зависимости активности бобров от прочих факторов
n <- names(df)
f <- as.formula(paste('class ~', paste(n[!n %in% 'class'], collapse = '+')))
#Создадим нейронную сеть по тренировочным данным, в которых 2 скрытых слоя (5 и 3)
d_net <- neuralnet(f, data = train_data, hidden = c(5, 3), linear.output = F)
plot(d_net)
#Посмотрим на результаты по тестовой выборке
predicted <- compute(d_net, test_data)
print(head(predicted$net.result))
#Округлим результаты
predicted$net.result <- sapply(predicted$net.result, round, digits = 0)
#После округления создадим таблицу сравнения результатов
test1 <- table(test_data$class, predicted$net.result)
test1
#По главной диагонали верные результаты, по минорной - ошибочные
#Посмотрим на количество 0 и 1 в тестовой выборке, чтобы проверить правильность
table(test_data$class)
#Оценим точность модели
Accuracy_nn <- (test1[1,1] + test1[2, 2])/sum(test1)
Accuracy_nn <- round(Accuracy_nn, digits = 2)
Accuracy_nn

#kohonen
install.packages("kohonen")
library('kohonen')
#Выделим искомый столбец в отдельную переменную
df2 <- df[1:8]
df2_a <- df[9]
#Далее зададим обучающую выборку - 559 строк (80%), 
#оставшися 140 будут использоваться в качестве тестовой выборки
train <- sample(nrow(df2), 559)
X_train <- scale(df2[train,])
X_test <- scale(df2[-train,],
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:center"))
train_data2 <- list(measurements = X_train,
                    df2_a = df2_a[train,])
test_data2 <- list(measurements = X_test,
                   df2_a = df2_a[-train,])
#Строим модель
mygrid <- somgrid(5, 5, 'hexagonal')
som.df2 <- supersom(train_data2, grid = mygrid) 
som.predict <- predict(som.df2, newdata = test_data2)
#Confusion matrix
test2 <- table(df2_a[-train,], som.predict$predictions$df2_a)
test2
#Оценим точность модели
Accuracy_k <- (test2[1,1] + test2[2, 2])/sum(test2)
Accuracy_k <- round(Accuracy_k, digits = 2)
Accuracy_k

#rsns
install.packages('RSNNS')
library(RSNNS)
df3 <- df[sample(1:nrow(df), length(1:nrow(df))), 1:ncol(df)]
df3_Values <- df3[, 1:8]
df3_Target <- df3[, 9]
#Разделим на тренировочную и тестовую выборки и нормализуем данные
df3 <- splitForTrainingAndTest(df3_Values, df3_Target, ratio = 0.2)
df3 <- normTrainingAndTestSet(df3)
#Строим модель
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
#Оценим точность модели
Accuracy_rsns <- (test3[1,1] + test3[2, 2])/sum(test3)
Accuracy_rsns <- round(Accuracy_rsns, digits = 2)
Accuracy_rsns

#Сводная таблица результатов
results <- cbind(Accuracy_nn, Accuracy_k, Accuracy_rsns)
rownames(results) <- c("Accuracy")
colnames(results) <- c("NeuralNet", "Kohonen", "RSNNS")
results
#Лучше справилась бибблиотека kohonen, точность составила 100%
#Библиотеки NeuralNet и RSNNS показали одинаковый результат - 93%
