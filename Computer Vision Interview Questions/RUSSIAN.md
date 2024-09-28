# **50 Computer Vision interview questions 2024**
## 1. What is Computer Vision, and how does it differ from image processing?
Компьютерное зрение - это область искусственного интеллекта, посвященная автоматической обработке визуальной информации. То есть на вход картинка/видео/облако точек, а на выходе может быть как та же модальность, так и совершенно новая, например: количество людей на картинке, описание событий на видео, карта глубины. Обработка изображений - это область компьютерного зрения, посвящённая увеличению "информативности" картинки, то есть - преобразованиям изображений с сохранением модальности: минмакс-нормализация, CLAHE, повышение чёткости, обнаружение краёв.
## 2. Can you explain the concept of feature extraction in Computer Vision?
Коротко: извлечение признаков -- это облегчающее процесс автоматической интерпретации приведение картинки (или любого другого экземпляра входной модальности) к набору признаков (другой модальности).
Например, если взять ResNet-50 (по слой Global Average Pooling), то выводом будет вектор длины 2048, который называют вектором признаков. Чем более различны картинки в тех смыслах, которые в модель закладывались в процессе обучения, тем больше расстояние между их векторами признаков.
## 3. What are some common applications of Computer Vision?
Распознавание объектов, повышение разрешения, определение сходства, OCR, V-SLAM. 
## 4. Describe the process of image segmentation.
Задача сегментации разделяется на три: семантическая, экземплярная и паноптическая. Все их объединяет формат входной информации (изображение) и общий вид вывода: каждому входному пикселю присваивается число (или набор чисел), обозначающее принадлежность этого пикселя какому-либо реальному объекту (для простоты дальнейшего изложения - фон тоже объект). В случае семантической сегментации выводом является сопоставление каждого входного пикселя одному из классов - это хорошо подходит для задачи определения размеров нефтяных разливов. Экземплярная сегментация сопоставляет каждый пиксель не только классу, но и конкретному экземпляру этого класса - это хорошо подходит, к примеру, для оценки размеров деревьев. А паноптическая сегментация суть объединение первых двух: для подсчитываемых объектов (например люди, монеты, бутылки) вывод совпадает с выводом экземплярной, а для остальных (снег, трава, зеркальная поверхность, лопнувшие капилляры) - с выводом семантической.
## 5. What is the purpose of edge detection in Computer Vision?
Обнаружение краёв применимо как облёгченный способ сегментации - когда искомый объект хорошо контрастирует с фоном и имеет чёткие края. Кроме того, методы обнаружения краёв пригодны для поиска ключевых точек - а значит, на них можно построить трекинг или 3D-реконструкцию.
## 6. Can you name some popular deep learning frameworks used in Computer Vision?
PyTorch, TensorFlow, PaddlePaddle
## 7. Explain the concept of Convolutional Neural Networks (CNNs) and their role in Computer Vision.
### The definition
Свёрточные нейросети - это нейросети, имеющие хотя бы один свёрточный слой. Но обо всём по порядку.
#### What is convolution in image processing?
В обработке изображений свёртка - это применение методом скользящего окна к входному тензору с шейпом (Ic, Ih, Iw) некоторого фильтра - тензора с шейпом (Ic, Fc, Fh, Fw). Результат каждого из "скольжений" - это сумма поэлементного произведения фильтра и попавшей в скользящее окно части входа. Все эти суммы располагаются в том же порядке, что и движение скользящего окна, а полученный тензор и есть результат свёртки, количество каналов в нём равно Fc.
#### What is convolutional layer?
Слой свёртки - это функция свёртки, фильтры которой принадлежат к весам модели, то есть числа в фильтрах обновляются в процессе обучения модели с целью снижения ошибки на обучающем наборе данных.
#### The motivation
Свёрточные слои:
1. Обеспечивают локальную стабильность, например: инвариантность вывода к положению объекта на картинке
2. Позволяют нейросети самой уточнять правила извлечения признаков из входа.
3. Требуют гораздо меньше параметров, чем полносвязные слои
### The impact
Именно свёрточные нейросети сделали глубокое обучение мейнстримом не только в компьютерном зрении, но в data science вообще - с AlexNet в 2012, который застакал 11 свёрточных слоёв для классификации изображений на датасете ImageNet, начинается эпоха глубокого обучения в компьютерном зрении - после этого все SOTA на этом датасете использовали свёрточные нейронки. В наше время два наиболее известных направления компьютерного зрения - это классическое CV и CV+DL, и второе из них почти всегда означает, что для решения будет использовано несколько свёрточных слоёв. 
## 8. What is object detection, and how is it different from object recognition?
Object detection task is to locate and classify object(s) on image.
According to my practice, the term “object recognition” is used synonymously these days, but back in the days when such a question may be asked in some decent place, it meant nothing but image classification.
## 9. Can you explain the terms precision and recall in the context of object detection?
These quality measures address two different types of errors
Precision=TP/(FP+TP), so it's 1 when everything the model predicts matches with Ground-Truth markup.
Recall=TP/(FN+TP), so it's 1 when each Ground-Truth has its own match among model predictions.
The most common match criteria is, besides class equivalence, is IoU, or Jaccard index - `intersection(gt,pred).area/union(gt,pred).area`, for example IoU>0.5 for a pair of same class is traditionally means that such prediction fits into the True Positive definition
## 10. What is the purpose of Non-Maximum Suppression (NMS) in object detection?
The purpose of this procedure is to avoid precision downfall by filtering out predictions that overlap too much. The criteria of overlapping is commonly IoU, and the value of IoU threshold is yet another object detection model hyperparameter - as it impacts precision and therefore the choose of best model. The NMS could be class agnostic.
## 11. How does image classification differ from image segmentation?
Классификация изображений - это сопоставление всего изображения конкретному классу (или набору классов, в случае multi-label classification). Сегментация изображения - это классификация каждого его пикселя в отдельности.
## 12. What is the purpose of image augmentation in deep learning for Computer Vision?
The definition of augmentation is getting a sample that differs from original but remain having critical features. The purposes are:
1. To increase model robustness by applying transforms it should be invariant to (negative, translate, rotate, scale, noise, blur, deformations, cutout)
2. To gain auxiliary train data at low cost (everything above + mixup, mosaic)
The subsequent purposes are: overfitting prevention and generalization improvement.
## 13. Describe the concept of transfer learning and its relevance in Computer Vision.

## 14. Explain the role of pooling layers in Convolutional Neural Networks.
## 15. What are the advantages and disadvantages of using CNNs for image classification?
## 16. Can you explain how data imbalance can affect the performance of a Computer Vision model?
## 17. What are some common techniques for reducing overfitting in deep learning models?
## 18. How does the concept of batch normalization help in training deep neural networks?
## 19. What is the purpose of data preprocessing in Computer Vision?
## 20. Describe the steps involved in training a Convolutional Neural Network.
## 21. Can you explain the concept of backpropagation and its role in training deep neural networks?
## 22. What is mean squared error (MSE), and how is it used in evaluating regression models in Computer Vision?
## 23. Explain the concept of object tracking in Computer Vision.
## 24. What are some challenges associated with real-time object detection?
## 25. How can you evaluate the performance of a Computer Vision model?
## 26. What is image registration, and how is it used in Computer Vision applications?
## 27. Can you explain the concept of image denoising?
## 28. What are some popular algorithms used for image feature extraction?
## 29. Describe the concept of histogram equalization and its applications in image processing.
## 30. What is optical character recognition (OCR), and how is it implemented in Computer Vision?
## 31. Can you explain the concept of semantic segmentation?
## 32. How does depth estimation work in Computer Vision?
## 33. What is the role of convolution in Convolutional Neural Networks?
## 34. Explain the concept of max-pooling and average pooling in CNNs.
## 35. What are some common activation functions used in deep learning for Computer Vision?
## 36. Describe the concept of image stitching and its applications.
## 37. What is the purpose of the softmax function in the output layer of a neural network?
## 38. Explain the concept of vanishing gradients in deep learning.
## 39. How does dropout regularization work in deep neural networks?
## 40. What are some techniques for handling occlusion in object detection?
## 41. Can you explain the concept of image pyramid in Computer Vision?
## 42. How does transfer learning help in training deep learning models with limited data?
## 43. What are some common metrics used for evaluating object detection algorithms?
## 44. Describe the concept of image inpainting.
## 45. What is the role of dilated convolutions in Convolutional Neural Networks?
## 46. Explain the concept of generative adversarial networks (GANs) and their applications in Computer Vision.
## 47. What are some common challenges faced in image classification tasks?
## 48. Can you explain the concept of data augmentation and its importance in training deep learning models?
## 49. Describe the concept of image super-resolution.
## 50. What are some emerging trends in Computer Vision research and applications?
## 