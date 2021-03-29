# Detecção de pinos e molas para o Projeto Springer - SALCOMP
O projeto tem como base principal usar o [PyTorch YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) a fim de detectar elementos presentes na fonte de alimentação do Springer. Estes elementos foram classificados como **Pino** e **Molas**

[YOLO](https://pjreddie.com/darknet/yolo/) (**You Only Look Once**) é um modelo de rede neural profunda o qual está otimizado para obter detecções em velocidades elevadas, isto é, torna-se uma boa opção para rodar em **VÍDEO**. O desempenho funciona muito melhor se o computador tem uma GPU dedicada da NVIDIA.

Por default este modelo foi treinado para dados customizados, ou seja, usando um **dataset** criado a partir de imagens obtidas do projeto em questão. O mesmo escontra-se em ```/data/custom```.

As etapas a seguir, tem como objetivo utilizar a detecção dos elementos do Springer, utilizando uma Webcam Logitech BRIO 4K. Para isso, usando o Anaconda ou Pycharm, cria-se um ambiente virtual no computador.

# Criação do ambiente no Pycharm
Cria-se un ambiente chamado ```springer_yolo``` o qual tem a versão 3.6 do interpretador **Python**.
``` 
python -n springer_yolo python=3.6
```

É necessário ativarmos o ambiente ```springer_yolo```, antes de instalarmos os requisitos principais para o funcionamento do algoritmo.
```
source activate springer_yolo
```

# Instalação dos pacotes necessários para o ambiente virtual
Dentro do arquivo ```requirements.txt``` encontram-se os pacotes necessários de acordo com a versão também. No terminal de comando do Pycharm, digite:
```
pip install -r requirements.txt
```

# Realizar o Download dos modelos treinados e pesos 
A fim de executar o modelo **Yolo** temos que realizar o download dos pesos da rede neural, os quais são os valores que possuem as conexões entre os neurônios da Rede neural principal do Yolo.  O modelo do Yolo pode ser pesado, por isso, é melhor criar o seu próprio dataset e treinar o modelo a partir dos novos dados.

```
bash weights/download_weights.sh
```

Movemos os pesos baixados na pasta chamada weights
```
mv yolov3.weights weights/
```

<!-- # Executar o detector de objetos em vídeo
Para a aplicação do Projeto Springer, podemos executar o script ```deteccao_video.py``` e dessa forma abrir a câmera de vídeo (webcam). Em ambientes linux, verifique em qual **device** a sua câmera está sendo reconhecida. Para isso, utilize o comando no terminal ```ls /dev/video*```.
```
python deteccao_video.py
```

# Notificações
Caso seja necessário rodar o script com algum vídeo do processo treinado sendo executado, digite o seguinte comando no terminal.

```
python deteccao_video.py --webcam 0 --diretorio_do_video <nome_do_arquivo.mp4>
``` -->
# Treinamento

Para o presente projeto deve-se criar um modelo com as classes que são necessárias, tais como: "pino e molas". Tome cuidado para que o dataset do próprio Yolo seja incluído, pois resultará em ruídos desnecessários para a solução. A seguir, são demonstrados como serão reaizados as etapas de treinamento e geração de **labels** customizado.

1. Reotule as imagens no formato YOLO. Para isso, instale o SW [labelImg](https://github.com/tzutalin/labelImg) para windows ou Linux. Caso dê algum tipo de problema que não permita criar as classes ou que não consiga excluir as classes já existentes, é possível instalar pelo terminal através da execução do seguinte comando:
```
pip install labelImg
```
2. Dentro do diretório ```/data/custom```, crie 2 diretórios chamados **images** e **labels**. As imagenes rotuladas devem no **data/custom/images** enquanto que o **metadata** tem que estar em **data/custom/labels**.
Para cada ```imagen.jpg``` deve de existir um arquivo imagen.txt (metadata com o mesmo nome da imagen)

3. Dentro do diretório ```config```, é necessário executar o arquivo **create_custom_model** para geral um arquivo **.cfg**, o qual contém a informação sobre a rede neural para executar as detecções. Deve-se informar também, o número de classes (no nosso caso são 2 -- pinos e molas). Veja o comando abaixo que deve ser executado.
```
cd config
bash create_custom_model.sh <Numero_de_classes_a_detectar>
cd ..
```

4. Agora é necessário fazemos o download da estrutura dos pesos do YOLO para transferir o **learning** nos pesos.

```
cd weights
bash download_darknet.sh
cd ..
```

5. O arquivo ```data/custom/classes.names``` deve conter o nome das classes de acordo como foram rotuladas.

6. Os arquivos ```data/custom/valid.txt``` e ```data/custom/train.txt``` devem conter o endereço de as imagens se encontram. Isto pode ser gerado com a execução do seguinte comando (considerando que as imagens estão dentro de ```data/custom/images```)
```
python split_train_val.py
```

## Execução do treinamento

 ```
 python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74 --batch_size 2
 ```

## Script de execução para detecção de molas e medidas entre os pinos em vídeo
```
python deteccion_video.py --model_def config/yolov3-custom.cfg --checkpoint_model checkpoints/yolov3_ckpt_99.pth --class_path data/custom/classes.names  --weights_path checkpoints/yolov3_ckpt_99.pth  --conf_thres 0.85
```

O threshold é um argumento de entrada que pode ser modificado conforme o nível de confiabilidade do treinamento, isto é, quanto maior o **Threshold** -- o nível de acerto será melhor. Ao mesmo tempo, isso significa que o **dataset** deve ser maior e isso requer uma maior rotulação de imagens.
