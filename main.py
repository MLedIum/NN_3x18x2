import random
from PIL import Image
import os
import numpy

# Задаем переменные
width = height = 3
train_photos = list()
epoch_for_notification = 100

# Получаем текущую директорию
path = os.path.dirname(os.path.abspath(__file__)) + "\\"
# И директорию с тренировочными фото
trainphotos_dir = path + "trainphotos\\"

# Получаем тренировочные файлы
for photo in os.listdir(trainphotos_dir):
    train_photos.append(trainphotos_dir + photo)


class NN:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, epochs):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.epochs = epochs

        # Задаем случайные значения весов 
        self.wf = numpy.random.rand(hiddennodes, inputnodes)
        self.ws = numpy.random.rand(outputnodes, hiddennodes)

        # Функция активации
        self.activation_function = lambda x: 1 / (1 + numpy.exp(-x))

    def train(self):
        epoch_n = 0
        for epoch in range(self.epochs):

            # Каждые epoch_for_notification эпох выводим текущую эпоху
            if epoch_n == epoch_for_notification:
                print(epoch)
                epoch_n = 0
            
            # Проходимся по тренировочным фото
            for photo in os.listdir(trainphotos_dir):

                # Загружаем фотографию
                pix = Image.open(trainphotos_dir + photo).load()

                colors = list()

                # Наполняем массив данными о цветах пикселей
                for x in range(width):
                    for y in range(height):
                        color = (pix[x, y][0] + pix[x, y][1] + pix[x, y][2]) / 256
                        colors.append(color)

                # Убираем лишние символы из названия файла
                photo_name = photo.split(".png")[0]

                # Создаем матрицу из входных значений
                input_array = numpy.array([colors], float).T
                # Получаем правильный ответ
                target = numpy.array(
                    [[photo_name.split("_")[0], photo_name.split("_")[1]]], float
                ).T

                # Ужножаем матрицу входных значений на веса
                hidden_inputs = numpy.dot(self.wf, input_array)
                # Функция активации нейронов скрытого слоя
                hidden_outputs = self.activation_function(hidden_inputs)

                # Ужножаем матрицу выходных значений нейронов скрытого слоя на веса
                final_inputs = numpy.dot(self.ws, hidden_outputs)
                # Функция активации нейронов выходного слоя
                final_output = self.activation_function(final_inputs)

                # Высчитываем ошибки
                output_errors = target - final_output
                hidden_errors = numpy.dot(self.ws.T, output_errors)

                # Изменяем веса
                self.ws += self.lr * numpy.dot(
                    (output_errors * final_output * (1.0 - final_output)),
                    numpy.transpose(hidden_outputs),
                )
                self.wf += self.lr * numpy.dot(
                    (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                    numpy.transpose(input_array),
                )

            # Показываем переход в следующую эпоху
            epoch_n += 1

    def query(self):

        # Открываем фотографию
        pix = Image.open(path + "photo.png").load()

        colors_q = list()

        # Наполняем массив цветами по порядку (0 - черный, 1 - белый)
        for x in range(width):
            for y in range(height):
                color = (pix[x, y][0] + pix[x, y][1] + pix[x, y][2]) / 765
                colors_q.append(color)

        # Создаем матрицу из входных значений
        inputs = numpy.array([colors_q], float).T

        hidden_inputs = numpy.dot(self.wf, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.ws, hidden_outputs)
        final_output = self.activation_function(final_inputs)

        if 0 < final_output[0] < 0.5:
            if 0 < final_output[1] < 0.5:
                print("Пустая картинка")
            elif 0.5 < final_output[1] < 1:
                print("На картинке есть горизонтальная линия")
        elif 0.5 < final_output[0] < 1:
            if 0 < final_output[1] < 0.5:
                print("На картинке есть вертикальная линия")
            elif 0.5 < final_output[1] < 1:
                print("На картинке есть вертикальная и горизонтальная линии")

# Указываем некоторые значения
inputnodes = 9 # кол-во входных нейронов
hiddennodes = 18 # кол-во скрытых нейронов
outputnodes = 2 # кол-во выходных нейронов
learningrate = 0.3  # коэф. обучения
epochs = 1000 # кол-во эпох обучения

# Создаем нейронную сеть
network = NN(inputnodes, hiddennodes, outputnodes, learningrate, epochs)
# Запускаем функцию тренировки
network.train()

# Бесконечный цикл 
while True:
    input("Нажмите Enter для сканирования photo.png")
    network.query()
