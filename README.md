# Detecção de Objetos - Utilizando Yolo v3

Este repositório é baseado no projeto [PyTorch YOLOv3](https://github.com/puigalex/deteccion-objetos-video) para executar a detecção de objetos no vídeo. Foi desenvolvido este projeto para adicionar a capacidade de detectar objetos em uma transmissão de vídeo ao vivo.

[YOLO](https://pjreddie.com/darknet/yolo/) (**You Only Look Once**) é um modelo que está otimizado para gerar detecções de elementos a uma velocidade muito alta, por isso é uma opção muito boa para usá-lo em vídeo. O treinamento e as previsões com este modelo se beneficiam de um computador que possui uma GPU NVIDIA.

Por padrão, este modelo é pré-treinado para detectar 80 objetos diferentes, a lista deles está no arquivo [data/coco.names](https://github.com/puigalex/deteccion-objetos-video/blob/master/data/coco.names)

Os passos a seguir para poder executar a detecção de objetos no vídeo de uma webcam são os seguintes (A criação do ambiente pressupõe que o Anaconda está instalado no computador):

# Criar ambiente
Para colocar nossos pacotes python em ordem, primeiro vamos criar um ambiente chamado `detecObj` que tenha a versão 3.6 do python
``` 
conda create -n detecObj python=3.6
```

Ativamos o ambiente `detecObj` para garantir que estamos no ambiente correto ao instalar todos os pacotes necessários
```
source activate detecObj
```

# Instalação das Dependências

Estando dentro de nosso ambiente vamos instalar todos os pacotes necessários para rodar nosso detector de objetos de vídeo, a lista de pacotes e versões a serem instaladas estão dentro do arquivo requirements.txt, então instalaremos referindo-se a esse arquivo
```
pip install -r requirements.txt
```

# Descarregando os pesos do modelo treinado

Para rodar o modelo YOLO teremos que baixar os pesos da rede neural, os pesos são os valores que todas as conexões têm entre os neurônios da rede neural YOLO, este tipo de modelos são computacionalmente muito pesados ​​para treinar do zero, baixando o modelo pré-treinado é uma boa opção.

```
bash weights/download_weights.sh
```

Nós movemos os pesos baixados para a pasta chamada `weights`
```
mv yolov3.weights weights/
```

# Correr el detector de objetos en video 
Por ultimo corremos este comando el cual activa la camara web para poder hacer deteccion de video sobre un video "en vivo"
```
python deteccion_video.py
```

# Modificaciones
Si en vez de correr detección de objetos sobre la webcam lo que quieres es correr el modelo sobre un video que ya fue pre grabado tienes que cambiar el comando para correr el codigo a:

```
python deteccion_video.py --webcam 0 --directorio_video <directorio_al_video.mp4>
```

# Entrenamiento 

Ahora, si lo que quieres es entrenar un modelo con las clases que tu quieras y no utilizar las 80 clases que vienen por default podemos entrenar nuestro propio modelo. Estos son los pasos que deberás seguir:

Primero deberás etiquetar las imagenes con el formato VOC, aqui tengo un video explicando como hacer este etiquetado: 

Desde la carpeta config correremos el archivo create_custom_model para generar un archivo .cfg el cual contiene información sobre la red neuronal para correr las detecciones
```
cd config
bash create_custom_model.sh <Numero_de_clases_a_detectar>
cd ..
```
Descargamos la estructura de pesos de YOLO para poder hacer transfer learning sobre esos pesos
```
cd weights
bash download_darknet.sh
cd ..
```

## Poner las imagenes y archivos de metadata en las carpetar necesarias

Las imagenes etiquetadas tienen que estar en el directorio **data/custom/images** mientras que las etiquetas/metadata de las imagenes tienen que estar en **data/custom/labels**.
Por cada imagen.jpg debe de existir un imagen.txt (metadata con el mismo nombre de la imagen)

El archivo ```data/custom/classes.names``` debe contener el nombre de las clases, como fueron etiquetadas, un renglon por clase.

Los archivos ```data/custom/valid.txt``` y ```data/custom/train.txt``` deben contener la dirección donde se encuentran cada una de las imagenes. Estos se pueden generar con el siguiente comando (estando las imagenes ya dentro de ```data/custom/images```)
```
python split_train_val.py
```

## Treinamento

```
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74 --batch_size 8
```

```
tensorboard --logdir='logs'
```

## Correr deteccion de objetos en video con nuestras clases
```
python deteccion_video.py --model_def config/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_99.pth --class_path data/custom/classes.names  --weights_path checkpoints/yolov3_ckpt_99.pth  --conf_thres 0.85
```

Criando TAG

```
git tag -a v1.1 -m "versaõ estável 1.1"
```


