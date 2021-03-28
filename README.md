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

# Executando o detector de objetos em vídeo
Finalmente, executamos este comando que ativa a câmera da web para ser capaz de detectar o vídeo em um vídeo "ao vivo".
```
python deteccion_video.py
```

# Modificações
Se, em vez de executar a detecção de objetos na webcam, o que você deseja é executar o modelo em um vídeo que já foi pré-gravado, você deve alterar o comando para executar o código para:

```
python deteccion_video.py --webcam 0 --directorio_video <directorio_al_video.mp4>
```

# Treinamento (Novo Dataset) 

Deve-se seguir as seguintes etapas paraa realizar treinamento com o próprio modelo:

Primeiro você deve rotular as imagens com o formato VOC, basicamento usando labelImg no formato yolo:

Na pasta de configuração, executaremos o arquivo `create_custom_model` para gerar um arquivo `.cfg` que contém informações sobre a rede neural para executar as detecções
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

## Coloque as imagens e os arquivos de metadados nas pastas necessárias

As imagens marcadas devem estar no diretório **data/custom/images** enquanto as tags / metadados da imagem devem estar em **data/custom/labels**.
Para cada imagem.jpg deve haver uma imagem.txt (metadados com o mesmo nome da imagem)

O arquivo ```data/custom/classes.names``` deve conter o nome das classes, como foram rotuladas, uma linha por classe.

Os arquivos ```data/custom/valid.txt``` e ```data/custom/train.txt``` devem conter o endereço onde cada uma das imagens está localizada. Estes podem ser gerados com o seguinte comando (com as imagens já dentro ```data/custom/images```)
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

## Executando a detecção de objetos em vídeo com nossas classes
```
python deteccion_video.py --model_def config/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_99.pth --class_path data/custom/classes.names  --weights_path checkpoints/yolov3_ckpt_99.pth  --conf_thres 0.85
```

Criando TAG

```
git tag -a v1.1 -m "versaõ estável 1.1"
```


